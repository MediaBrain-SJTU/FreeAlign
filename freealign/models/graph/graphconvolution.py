import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional, List, Union

class Gconv(nn.Module):
    r"""
    Graph Convolutional Layer which is inspired and developed based on Graph Convolutional Network (GCN).
    Inspired by `Kipf and Welling. Semi-Supervised Classification with Graph Convolutional Networks. ICLR 2017.
    <https://arxiv.org/abs/1609.02907>`_

    :param in_features: the dimension of input node features
    :param out_features: the dimension of output node features
    """
    def __init__(self, in_features: int, out_features: int):
        super(Gconv, self).__init__()
        self.num_inputs = in_features
        self.num_outputs = out_features
        self.a_fc = nn.Linear(self.num_inputs, self.num_outputs)
        self.u_fc = nn.Linear(self.num_inputs, self.num_outputs)

    def forward(self, A: Tensor, x: Tensor, norm: bool=True) -> Tensor:
        r"""
        Forward computation of graph convolution network.

        :param A: :math:`(b\times n\times n)` {0,1} adjacency matrix. :math:`b`: batch size, :math:`n`: number of nodes
        :param x: :math:`(b\times n\times d)` input node embedding. :math:`d`: feature dimension
        :param norm: normalize connectivity matrix or not
        :return: :math:`(b\times n\times d^\prime)` new node embedding
        """
        if norm is True:
            A = F.normalize(A, p=1, dim=-2)
        ax = self.a_fc(x)
        ux = self.u_fc(x)
        x = torch.bmm(A, F.relu(ax)) + F.relu(ux) # has size (bs, N, num_outputs)
        return x



class Siamese_Gconv(nn.Module):
    r"""
    Siamese Gconv neural network for processing arbitrary number of graphs.

    :param in_features: the dimension of input node features
    :param num_features: the dimension of output node features
    """
    def __init__(self, in_features, num_features):
        super(Siamese_Gconv, self).__init__()
        self.gconv = Gconv(in_features, num_features)

    def forward(self, g1: Tuple[Tensor, Tensor, Tensor, int], *args) -> Union[Tensor, List[Tensor]]:
        r"""
        Forward computation of Siamese Gconv.

        :param g1: The first graph, which is a tuple of (:math:`(b\times n\times n)` {0,1} adjacency matrix,
         :math:`(b\times n\times d)` input node embedding, normalize connectivity matrix or not)
        :param args: Other graphs
        :return: A list of tensors composed of new node embeddings :math:`(b\times n\times d^\prime)`
        """
        # embx are tensors of size (bs, N, num_features)
        emb1 = self.gconv(*g1)
        if len(args) == 0:
            return emb1
        else:
            returns = [emb1]
            for g in args:
                returns.append(self.gconv(*g))
            return returns