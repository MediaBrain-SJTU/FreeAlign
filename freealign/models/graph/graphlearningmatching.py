import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional, List, Union
from freealign.models.graph.graphconvolution import *
from freealign.models.graph.Sinkhorn import Sinkhorn
# from freealign.models.graph.affinity_layer import Affinity
import torch.optim as optim
# import torch_geometric.nn as pyg_nn
import dgl
from dgl.nn.pytorch import EGNNConv
# from opencood.utils.box_utils import project_box3d
# from opencood.utils.common_utils import convert_format, compute_iou

# class GraphLearningMatching(nn.Module):
#     def __init__(self, opt) -> None:
#         super().__init__()
#         self.sinkhorn = Sinkhorn(max_iter=opt.PCA.SK_ITER_NUM, epsilon=opt.PCA.SK_EPSILON, tau=opt.PCA.SK_TAU)
#         self.l2norm = nn.LocalResponseNorm(opt.PCA.FEATURE_CHANNEL * 2, alpha=opt.PCA.FEATURE_CHANNEL * 2, beta=0.5, k=0)
#         self.gnn_layer = opt.PCA.GNN_LAYER
#         #self.pointer_net = PointerNet(opt.PCA.GNN_FEAT, opt.PCA.GNN_FEAT // 2, alpha=opt.PCA.VOTING_ALPHA)
#         for i in range(self.gnn_layer):
#             if i == 0:
#                 gnn_layer = Siamese_Gconv(opt.PCA.FEATURE_CHANNEL * 2, opt.PCA.GNN_FEAT)
#             else:
#                 gnn_layer = Siamese_Gconv(opt.PCA.GNN_FEAT, opt.PCA.GNN_FEAT)
#             self.add_module('gnn_layer_{}'.format(i), gnn_layer)
#             self.add_module('affinity_{}'.format(i), Affinity(opt.PCA.GNN_FEAT))
#             if i == self.gnn_layer - 2:  # only second last layer will have cross-graph module
#                 self.add_module('cross_graph_{}'.format(i), nn.Linear(opt.PCA.GNN_FEAT * 2, opt.PCA.GNN_FEAT))
#         self.cross_iter = opt.PCA.CROSS_ITER
#         self.cross_iter_num = opt.PCA.CROSS_ITER_NUM
#         self.rescale = opt.PROBLEM.RESCALE

#     def forward(self, graph1, graph2):
#         ss = []
#         if not self.cross_iter:
#             # Vanilla PCA-GM
#             for i in range(self.gnn_layer):
#                 gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
#                 emb1, emb2 = gnn_layer([A_src, emb1], [A_tgt, emb2]) 
#                 affinity = getattr(self, 'affinity_{}'.format(i))
#                 s = affinity(emb1, emb2)
#                 s = self.sinkhorn(s, ns_src, ns_tgt, dummy_row=True)

#                 ss.append(s)

#                 if i == self.gnn_layer - 2:
#                     cross_graph = getattr(self, 'cross_graph_{}'.format(i))
#                     new_emb1 = cross_graph(torch.cat((emb1, torch.bmm(s, emb2)), dim=-1))
#                     new_emb2 = cross_graph(torch.cat((emb2, torch.bmm(s.transpose(1, 2), emb1)), dim=-1))
#                     emb1 = new_emb1
#                     emb2 = new_emb2
#         else:
#             # IPCA-GM
#             for i in range(self.gnn_layer - 1):
#                 gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
#                 emb1, emb2 = gnn_layer([A_src, emb1], [A_tgt, emb2])

#             emb1_0, emb2_0 = emb1, emb2
#             s = torch.zeros(emb1.shape[0], emb1.shape[1], emb2.shape[1], device=emb1.device)

#             for x in range(self.cross_iter_num):
#                 i = self.gnn_layer - 2
#                 cross_graph = getattr(self, 'cross_graph_{}'.format(i))
#                 emb1 = cross_graph(torch.cat((emb1_0, torch.bmm(s, emb2_0)), dim=-1))
#                 emb2 = cross_graph(torch.cat((emb2_0, torch.bmm(s.transpose(1, 2), emb1_0)), dim=-1))

#                 i = self.gnn_layer - 1
#                 gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
#                 emb1, emb2 = gnn_layer([A_src, emb1], [A_tgt, emb2])
#                 affinity = getattr(self, 'affinity_{}'.format(i))
#                 s = affinity(emb1, emb2)
#                 s = self.sinkhorn(s, ns_src, ns_tgt, dummy_row=True)
#                 ss.append(s)

#         data_dict.update({
#             'ds_mat': ss[-1],
#             'perm_mat': hungarian(ss[-1], ns_src, ns_tgt)
#         })
#         return data_dict


class GraphSAGEWithEdgeFeatures(nn.Module):
    def __init__(self, in_channels, out_channels, edge_dim):
        super(GraphSAGEWithEdgeFeatures, self).__init__()
        self.lin1 = nn.Linear(in_channels + edge_dim, out_channels)
        self.lin2 = nn.Linear(out_channels, out_channels)
        self.act = nn.ReLU()

    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index
        x_col = x[col]
        x_row = x[row]
        edge_feature = edge_attr[row, col]
        out = torch.cat([x_row, edge_feature], dim=-1)
        out = self.lin1(out)
        out = self.act(out)
        out = self.lin2(out)
        out = self.act(out)
        out = pyg_nn.global_add_pool(out, col)
        return out


class GNNModel(nn.Module):
    def __init__(self, opt) -> None:
        super().__init__()
        self.node_embedding_linear = nn.Linear(1, opt.input_dim)
        self.edge_embedding_linear = nn.Linear(1, opt.edge_dim)
        self.conv1 = GraphSAGEWithEdgeFeatures(opt.input_dim, opt.hidden_dim, opt.edge_dim)
        self.conv2 = GraphSAGEWithEdgeFeatures(opt.hidden_dim, opt.output_dim, opt.edge_dim)

    def forward(self, x, edge_attr):
        x = torch.relu(self.node_embedding_linear(x.cuda().unsqueeze(-1)))
        edge_attr = torch.relu(self.edge_embedding_linear(edge_attr))
        edge_index = self.get_fc_edge_index(x.shape[0])
        x = self.conv1(x, edge_index, edge_attr)
        x = self.conv2(x, edge_index, edge_attr)
        return x
    
    def get_fc_edge_index(self, num_nodes):
        edge_index = torch.stack([torch.tensor([i, j]).long() for i in range(num_nodes) for j in range(num_nodes)], dim = 1)
        return edge_index.cuda()
    
class ContrastiveLoss(nn.Module):
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, match_pair, similarity_matrix):
        # match_pair = [pair.long() for pair in match_pair]
        # anchor_1 = ego[match_pair[0]]
        # positive_1 = ego[match_pair[1]]
        # negative_list_1 = []
        # for i in range(len(edge)):
        #     if i not in match_pair[1]:
        #         negative_list_1.append(i)
        # negative_1 = edge[negative_list_1]
        # pos_distance_1 = torch.norm(anchor_1 - positive_1, dim=-1)
        # neg_distance_1 = torch.norm(anchor_1 - negative_1, dim=-1)
        # loss_1 = torch.relu(pos_distance_1 - neg_distance_1 + self.margin)

        # anchor_2 = edge[match_pair[1]]
        # positive_2 = ego[match_pair[0]]
        # negative_list_2 = []
        # for i in range(len(ego)):
        #     if i not in match_pair[0]:
        #         negative_list_2.append(i)
        # negative_2 = ego[negative_list_2]
        # pos_distance = torch.norm(anchor_2 - positive_2, dim=-1)
        # neg_distance = torch.norm(anchor_2 - negative_2, dim=-1)
        # loss = torch.relu(pos_distance - neg_distance + self.margin)
        # loss = 1 - similarity_matrix[match_pair[0], match_pair[1]].mean()
        # similarity_matrix[match_pair[0], match_pair[1]] = 0
        # loss += similarity_matrix.mean()/5
        positive_mean = similarity_matrix[match_pair[0], match_pair[1]].mean()
        negative_mean = similarity_matrix.mean()
        loss = negative_mean - positive_mean + self.margin

        return loss
    
class GraphLearningMatching(nn.Module):
    def __init__(self, opt) -> None:
        super().__init__()
        self.gnn = GNNModel(opt)
        self.loss = ContrastiveLoss(opt.margin)
        self.match = Sinkhorn()

    def forward(self, batch_data):
        graph_embedding_list = [[self.gnn(batch_data['scores'][batch][i], batch_data['graphs'][batch][i]) for i in range(len(batch_data['graphs'][batch]))] for batch in range(len(batch_data['scores']))]
        ego_graph_embedding = graph_embedding_list[0][0]
        loss = 0
        loss_item = 0
        match_pair_sum = sum([len(batch_data['match_pair'][batch]) for batch in range(len(batch_data['scores']))])
        if match_pair_sum == 0:
            return 0
        for batch in range(len(batch_data['scores'])):
            for i in range(1, len(graph_embedding_list[batch])):
                similarity_matrix = F.cosine_similarity(ego_graph_embedding.unsqueeze(1), graph_embedding_list[batch][i].unsqueeze(0), dim = -1)
                print(similarity_matrix)
                print(batch_data['match_pair'][batch][i - 1])
                # match_matching_matrix = torch.bmm(ego_graph_embedding.unsqueeze(0), graph_embedding_list[i].unsqueeze(0).transpose(1, 2))
                loss_item = self.loss(batch_data['match_pair'][batch][i - 1], similarity_matrix)
                # match_pair = self.match(similarity_matrix, len(ego_graph_embedding), len(graph_embedding_list[0][0]), False)
                # loss_2 = self.loss(batch_data['match_pair'][batch][i - 1], ego_graph_embedding, graph_embedding_list[batch][i])
                
        loss += loss_item
        loss /= len(batch_data['scores'])


        return loss
    
def cal_accurate(match_pair, boundingbox_pair):
    for i in range(len(match_pair)):
        if match_pair[i][0] in boundingbox_pair:
            if match_pair[i][1] in boundingbox_pair[match_pair[i][0]]:
                return 1



class EGNN(nn.Module):
    def __init__(self, input_dim, hiddn_dim, out_dim):
        super().__init__()
        self.conv1 = EGNNConv(in_size=input_dim, hidden_size=hiddn_dim, out_size=hiddn_dim)
        self.conv2 = EGNNConv(in_size=hiddn_dim, hidden_size=hiddn_dim, out_size=out_dim)

    def forward(self, graph, node_feature, node_coord):
        h, x = self.conv1(graph, node_feature, node_coord)
        h, x = self.conv2(graph, h, x)
        return x