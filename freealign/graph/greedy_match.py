import torch
import numpy as np
from opencood.utils.box_utils import project_box3d
from opencood.utils.pose_utils import generate_noise
from opencood.utils.transformation_utils import tfm_to_pose, pose_to_tfm


import numpy as np
import torch
from opencood.utils.box_utils import project_box3d
from opencood.utils.pose_utils import generate_noise
from opencood.utils.transformation_utils import tfm_to_pose, pose_to_tfm
from optimize import optimize, optimize_rt, optimize_delta_theta
from itertools import product
# from freealign.optimize import optimize_rt
import os

def add_rot_noise(clean_tfm):
    pose = tfm_to_pose(clean_tfm)
    pose += generate_noise(0, 0.0)
    tfm = pose_to_tfm(np.expand_dims(pose, 0))
    return tfm.squeeze()
    
def get_best_match(node_i_ego, node_j_edge, max_error = 0.5, k = 1):
    distance_raw = np.abs(node_i_ego[:, np.newaxis] - node_j_edge)
    distance = distance_raw[:,:, 0] + k * distance_raw[:,:, 1]
    error = 100
    match = []
    for i in range(min(len(node_i_ego), len(node_j_edge))):
        error_item = np.min(distance)
        if error_item < max_error:
            min_index = np.argmin(distance)  # Returns the index of the flattened array
            min_index_2d = np.unravel_index(min_index, distance.shape)
            error += error_item
            distance[min_index_2d[0]] = 999
            distance[:,min_index_2d[1]] = 999
            match.append(min_index_2d)
        else:
            break
    if len(match) == 0:
        match = [0]
    return match, error/len(match)

def greedy_find_matching(node_i_ego, graph_edge, i, min_nodes=3):
    match_list = {}
    best_avg_err = 100
    best_matching_ever = []
    for j in range(graph_edge.shape[0]):
        node_j_edge = graph_edge[j]
        best_matching, avg_error = get_best_match(node_i_ego, node_j_edge)
        # best_matching.append((i,j))
        # match_list.append({'match': best_matching, 'err': avg_error})

        if len(best_matching) >= min_nodes:
            if avg_error <= best_avg_err:
                best_matching_ever = best_matching
                best_avg_err = avg_error
    return best_matching_ever, best_avg_err

def find_common_subgraph(graph1, graph2, min_nodes=3, error=0.3):
    best_error = 100
    best_match = []
    for i in range(graph1.shape[0]):
        match, error = greedy_find_matching(graph1[i], graph2, i, min_nodes)
        if len(match) > min_nodes:
            if error <= best_error:
                best_match = match
                best_error = error
    return len(best_match), best_match, best_error
# Test the function with two example graphs

def coor_trans_new(single_pred):
    graphs = []
    for cav in range(len(single_pred['pose'])): 
        pose = single_pred['pose'][cav]
        A = torch.zeros((pose.shape[0], pose.shape[0], 2))
        for i in range(pose.shape[0]):
            # A[i] = rotate_points_along_z_2d(-pose[i,:2].unsqueeze(0) + pose[:,:2], -single_pred['pose'][cav][i][-1].unsqueeze(0))
            A[i,:,0] = torch.norm(-pose[i,:2].unsqueeze(0) + pose[:,:2],dim=1)
            A[i,:,1] = (pose[i,-1] - pose[:,-1])
        graphs.append(np.array(A))
    return graphs

def coor_trans(box, tensor, transform):
    for cav in range(2):
        if cav == 0:
            noisy_t_matrix = np.identity(4)
        else:
            noisy_t_matrix = transform
        noisy_t_matrix[0:3,-1] -= noisy_t_matrix[0:3,-1]
        # if cav != 0:
        #     noisy_t_matrix = add_rot_noise(noisy_t_matrix)
        clean_box_ego = project_box3d(box[cav], transform)
        box_ego = project_box3d(box[cav], noisy_t_matrix)
        for i in range(box_ego.shape[0]):
            for j in range(box_ego.shape[0]):
                x_i = box_ego[i,:,0].mean(0)
                x_j = box_ego[j,:,0].mean(0)
                y_i = box_ego[i,:,1].mean(0)
                y_j = box_ego[j,:,1].mean(0)
                tensor[cav][i,j,0] = ((x_i) - (x_j)) / 2
                tensor[cav][i,j,1] = ((y_i) - (y_j)) / 2
    return tensor, clean_box_ego