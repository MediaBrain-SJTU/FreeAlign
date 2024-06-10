

from copy import deepcopy
from pathlib import Path
from typing import List, Tuple

import torch
from torch import nn
import torch.multiprocessing as mp

def cat_list(data: list) -> torch.Tensor:
    # input agent: [Agent1, Agent2, Agent3...]
    # output batch: BA*C*H*W
    tensor_list = []
    for batch in data:
        for agent in batch:
            tensor_list.append(agent)
    batch = torch.cat(tensor_list, dim=0)
    return batch

def batch_to_agent(batch: torch.Tensor, agent_num_list: list) -> list:
    # input batch: BA*C*H*W
    # output agent: [Agent1, Agent2, Agent3...]
    agent_list = []
    start = 0
    for agent_num in agent_num_list:
        agent_list.append(batch[start: start + agent_num])
        start += agent_num
    return agent_list

def MLP(channels: List[int], do_bn: bool = True) -> nn.Module:
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def normalize_keypoints(kpts, image_shape):
    """ Normalize keypoints locations based on image image_shape"""
    batch, agent, channel, height, width = image_shape
    one = kpts.new_tensor(1)
    size = torch.stack([one*width, one*width, one*height, one*height])[None]
    center = size / 2
    scaling = size.max(1, keepdim=True).values * 0.7
    return (kpts - center[:, None, :]) / scaling[:, None, :]


class KeypointEncoder(nn.Module):
    """ Joint encoding of visual appearance and location using MLPs"""
    def __init__(self, feature_dim: int, layers: List[int]) -> None:
        super().__init__()
        self.encoder = MLP([5] + layers + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts, scores):
        inputs = [kpts.transpose(1, 2), scores.unsqueeze(0).unsqueeze(0)]
        return self.encoder(torch.cat(inputs, dim=1))


def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob


class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, _ = attention(query, key, value)
        return self.merge(x.contiguous().view(batch_dim, self.dim*self.num_heads, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim*2, feature_dim*2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))


class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: List[str]) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalPropagation(feature_dim, 4)
            for _ in range(len(layer_names))])
        self.names = layer_names

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
        for layer, name in zip(self.layers, self.names):
            if name == 'cross':
                src0, src1 = desc1, desc0
            else:  # if name == 'self':
                src0, src1 = desc0, desc1
            delta0, delta1 = layer(desc0, src0), layer(desc1, src1)
            desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
        return desc0, desc1

def log_sinkhorn_iterations(Z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores: torch.Tensor, alpha: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z


def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1


class SuperBEVGlue(nn.Module):
    """SuperGlue feature matching middle-end

    Given two sets of keypoints and locations, we determine the
    correspondences by:
      1. Keypoint Encoding (normalization + visual feature and location fusion)
      2. Graph Neural Network with multiple self and cross-attention layers
      3. Final projection layer
      4. Optimal Transport Layer (a differentiable Hungarian matching algorithm)
      5. Thresholding matrix based on mutual exclusivity and a match_threshold

    The correspondence ids use -1 to indicate non-matching points.

    Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. SuperGlue: Learning Feature Matching with Graph Neural
    Networks. In CVPR, 2020. https://arxiv.org/abs/1911.11763

    """
    default_config = {
        'descriptor_dim': 256,
        'weights': 'indoor',
        'keypoint_encoder': [32, 64, 128, 256],
        'GNN_layers': ['self', 'cross'] * 9,
        'sinkhorn_iterations': 100,
        'match_threshold': 0.2,
        'resume': False,
        'val': False
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}

        self.kenc = KeypointEncoder(
            self.config['descriptor_dim'], self.config['keypoint_encoder'])

        self.gnn = AttentionalGNN(
            feature_dim=self.config['descriptor_dim'], layer_names=self.config['GNN_layers'])

        self.final_proj = nn.Conv1d(
            self.config['descriptor_dim'], self.config['descriptor_dim'],
            kernel_size=1, bias=True)

        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)

        if self.config['resume'] or self.config['val']:
            assert self.config['weights'] in ['indoor', 'outdoor']
            path = Path(__file__).parent
            path = path / 'weights/superglue_{}.pth'.format(self.config['weights'])
            self.load_state_dict(torch.load(str(path)))
            print('Loaded SuperGlue model (\"{}\" weights)'.format(
                self.config['weights']))

    def gnn_part(self, rank, input_data, output_queue):
        desc0, desc1 = input_data
        output = self.gnn(desc0, desc1)
        output_queue.put((rank, output))

    def forward(self, para):
        """Run SuperGlue on a pair of keypoints and descriptors"""
        # desc0, desc1 = data['descriptors0'], data['descriptors1']
        # kpts0, kpts1 = data['keypoints0'], data['keypoints1']
        desc0, desc1, kpts0, kpts1, score0, score1, shape0, shape1, match_pair = para
        desc0 = desc0.transpose(1,2)
        desc1 = desc1.transpose(1,2)

        if kpts0.shape[1] == 0 or kpts1.shape[1] == 0:  # no keypoints
            shape0, shape1 = kpts0.shape[:-1], kpts1.shape[:-1]
            return {
                'matches0': kpts0.new_full(shape0, -1, dtype=torch.int),
                'matches1': kpts1.new_full(shape1, -1, dtype=torch.int),
                'matching_scores0': kpts0.new_zeros(shape0),
                'matching_scores1': kpts1.new_zeros(shape1),
            }

        # Keypoint normalization.
        kpts0 = normalize_keypoints(kpts0, (1,1,1,100,352))
        kpts1 = normalize_keypoints(kpts1, (1,1,1,100,352))

        # Keypoint MLP encoder.
        flag0 = False
        flag1 = False
        if desc0.shape[-1] <= 1:
            kpts0 = kpts0.repeat(1,2,1)
            desc0 = desc0.repeat(1,1,2)
            score0 = score0.repeat(2)
            flag0 = True
        if desc1.shape[-1] <= 1:
            kpts1 = kpts1.repeat(1,2,1)
            desc1 = desc1.repeat(1,1,2)
            score1 = score1.repeat(2)
            flag1 = True

        # print(kpts0)
        desc0 = desc0 + self.kenc(kpts0, score0)
        # print(desc0)
        desc1 = desc1 + self.kenc(kpts1, score1)



        # Multi-layer Transformer network.
        desc0, desc1 = self.gnn(desc0, desc1)
        # Final MLP projection.
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
        if flag0:
            mdesc0 = mdesc0[:,:,0:1]
        if flag1:
            mdesc1 = mdesc1[:,:,0:1]
        # Compute matching descriptor distance.
        scores = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
        scores = scores / self.config['descriptor_dim']**.5
        # Run the optimal transport.
        scores = log_optimal_transport(
            scores, self.bin_score,
            iters=self.config['sinkhorn_iterations'])

        # Get the matches with score above "match_threshold".
        max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
        indices0, indices1 = max0.indices, max1.indices
        mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
        mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
        zero = scores.new_tensor(0)
        mscores0 = torch.where(mutual0, max0.values.exp(), zero)
        mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
        valid0 = mutual0 & (mscores0 > self.config['match_threshold'])
        valid1 = mutual1 & valid0.gather(1, indices1)
        indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))

        loss = []
        match_pair = self.modify_match_pair(match_pair, desc0.shape[-1], desc1.shape[-1])
        for i in range(len(match_pair[0])):
            x, y = match_pair[i]
            loss.append(-torch.log(scores[0][x][y].exp())) # check batch size == 1 ?
        # for p0 in unmatched0:
        #     loss += -torch.log(scores[0][p0][-1])
        # for p1 in unmatched1:
        #     loss += -torch.log(scores[0][-1][p1])
        loss_mean = torch.mean(torch.stack(loss))
        loss_mean = torch.reshape(loss_mean, (1, -1))


        return {
            'matches0': indices0, # use -1 for invalid match
            'matches1': indices1, # use -1 for invalid match
            'matching_scores0': mscores0,
            'matching_scores1': mscores1,
            'loss': loss_mean
        }

    def modify_match_pair(self, match_pair, N, M):
        agent0_list = []
        agent1_list = []
        for pair in match_pair:
            agent0_list.append(pair[0])
            agent1_list.append(pair[1])
        for i in range(N):
            if i not in agent0_list:
                match_pair.append((i, -1))
        for i in range(M):
            if i not in agent1_list:
                match_pair.append((-1, i))
        return match_pair
    
def split_data(data, num_agent, box_num):
    bs = len(num_agent)
    split_data = []
    load_list = []
    count = 0
    for batch in range(bs):
        batch_list = []
        for agent in range(num_agent[batch]):
            if agent == 0:
                ego_feature = data['descriptor'][:, :, count:count+box_num[batch][agent]]
                ego_kpts = data['boxes'][:, :, count:count+box_num[batch][agent]]
            else:
                batch_list.append([ego_feature.clone(), data['descriptor'][:, :, count:count+box_num[batch][agent]], ego_kpts, data['boxes'][:, :, count:count+box_num[batch][agent]]])
                load_list.append(batch_list[-1])
            count += box_num[batch][agent]
        split_data.append(batch_list)
    
    return split_data, load_list