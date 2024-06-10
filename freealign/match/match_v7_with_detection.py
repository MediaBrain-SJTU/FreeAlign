import numpy as np
import torch
from opencood.utils.box_utils import project_box3d
from opencood.utils.pose_utils import generate_noise
from opencood.utils.transformation_utils import tfm_to_pose, pose_to_tfm
from freealign.optimize import optimize, optimize_rt, optimize_delta_theta

import os
import copy
import cv2
from tqdm import tqdm
import argparse

class Opt:
    def __init__(self, min_anchors, anchor_error, box_error) -> None:
        self.min_anchors = min_anchors
        self.anchor_error = anchor_error
        self.box_error = box_error
        
def freealign_training(box_list1, box_list2, gt_pose_ego, gt_pose_agent, min_anchors=3, anchor_error=0.3, box_error=0.5):
    opt = Opt(min_anchors, anchor_error, box_error)
    if len(box_list1) <= 2 or len(box_list2) <= 2:
        return np.zeros(6)
    box_array1 = np.stack([np.stack([np.array(box_list1[i][j]) for j in range(len(box_list1[i]))], axis = 0) for i in range(len(box_list1))], axis = 0)
    box_array2 = np.stack([np.stack([np.array(box_list2[i][j]) for j in range(len(box_list2[i]))], axis = 0) for i in range(len(box_list2))], axis = 0)
    box = [box_array1, box_array2]

    tensor = [np.zeros((box[0].shape[0], box[0].shape[0], 1)), np.zeros((box[1].shape[0], box[1].shape[0], 1))]

    tensor, _ = coor_trans(box, tensor, np.stack([np.identity(4), np.identity(4)], axis=0))
    mcs_size, matched = find_common_subgraph(tensor[0], tensor[1], opt)
    trans_matrix = np.identity(3)
    Six_DoF_Pose = np.zeros(6)
    if mcs_size >= 3:
        R_estimated, T_estimated = optimize_rt(matched, box_ego = box[0], box_edge = box[1])
        trans_matrix[0:2,0:2] = R_estimated
        trans_matrix[0:2,2] = T_estimated
        Three_DoF_Pose = affine_matrix_to_3dof_pose(trans_matrix)
        Six_DoF_Pose[[0,1,4]] = Three_DoF_Pose
    else:
        return np.zeros((6))
    
    t_matrix = torch.linalg.solve(torch.tensor(gt_pose_ego), torch.tensor(gt_pose_agent))
    if (torch.tensor(T_estimated) - t_matrix[0:2,3]).mean() > 4:
        print(T_estimated, t_matrix[0:2,3])
        return np.zeros((6))
    # if error_t.mean() > 1 or error_r:
    #     return np.zeros((6))
    return Six_DoF_Pose

def get_pose_rotation(box_list1, box_list2, min_anchors=3, anchor_error=0.3, box_error=0.5):
    opt = Opt(min_anchors, anchor_error, box_error)
    if len(box_list1) <= 2 or len(box_list2) <= 2:
        return np.zeros(6)
    box_array1 = np.stack([np.stack([np.array(box_list1[i][j]) for j in range(len(box_list1[i]))], axis = 0) for i in range(len(box_list1))], axis = 0)
    box_array2 = np.stack([np.stack([np.array(box_list2[i][j]) for j in range(len(box_list2[i]))], axis = 0) for i in range(len(box_list2))], axis = 0)
    box = [box_array1, box_array2]

    tensor = [np.zeros((box[0].shape[0], box[0].shape[0], 1)), np.zeros((box[1].shape[0], box[1].shape[0], 1))]

    # tensor, box[1] = coor_trans(box, tensor, transform)
    tensor, _ = coor_trans(box, tensor, np.stack([np.identity(4), np.identity(4)], axis=0))
    # tensor, _ = coor_trans(box, tensor, transform)
    # tensor = coor_trans_new(single_pred)
    mcs_size, matched = find_common_subgraph(tensor[0], tensor[1], opt)
    trans_matrix = np.identity(3)
    Six_DoF_Pose = np.zeros(6)
    if mcs_size >= 3:
        R_estimated, T_estimated = optimize_rt(matched, box_ego = box[0], box_edge = box[1])
        trans_matrix[0:2,0:2] = R_estimated
        trans_matrix[0:2,2] = T_estimated
        Three_DoF_Pose = affine_matrix_to_3dof_pose(trans_matrix)
        Six_DoF_Pose[[0,1,4]] = Three_DoF_Pose
    else:
        return np.zeros((6))
    return Six_DoF_Pose

def affine_matrix_to_3dof_pose(matrix):
    # 提取平移向量
    translation = matrix[:2, 2]

    # 提取旋转矩阵
    rotation_matrix = matrix[:2, :2]

    # 计算旋转角度（单位：弧度）
    theta = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

    # 将旋转角度转换为度数
    theta_degrees = np.degrees(theta)

    # 返回3DoF位姿
    return np.array([translation[0], translation[1], theta_degrees])


def add_rot_noise(clean_tfm):
    pose = tfm_to_pose(clean_tfm)
    pose += generate_noise(0, 0.0)
    tfm = pose_to_tfm(np.expand_dims(pose, 0))
    return tfm.squeeze()

def judge_box(graph1, graph2, anchors, min_index_2d, max_error = 0.2):
    for i, anchor in enumerate(anchors):
        if np.abs(graph1[anchor[0]][min_index_2d[0]] - graph2[anchor[1]][min_index_2d[1]]) < max_error:
            continue
        else:
            return False
    return True
    

def find_anchors(distance_raw, max_error=0.1, graph1=None, graph2=None):
    
    distance = copy.deepcopy(distance_raw)
    best_error = 100
    intial_index = np.argmin(distance)  # Returns the index of the flattened array
    intial_index_2d = np.unravel_index(intial_index, distance.shape)
    best_anchors = [intial_index_2d]
    distance[intial_index_2d] = 999
    distance_bak = copy.deepcopy(distance)

    candidates = np.where(distance < max_error)
    candidates_list = [(candidates[0][i], candidates[1][i]) for i in range(len(candidates[0]))]

    for i, candidate in enumerate(candidates_list):
        distance = copy.deepcopy(distance_bak)
        error = distance[candidate]
        distance[candidate[0]] = 999
        distance[:,candidate[1]] = 999
        anchors = [intial_index_2d, candidate]
        for anchor in candidates_list:
            if anchor == candidate:
                continue
            else:
                if (distance[anchor] < max_error) and (judge_box(graph1, graph2, anchors, anchor, max_error*2)):
                    anchors.append(anchor)
                    distance[anchor[0]] = 999
                    distance[:,anchor[1]] = 999
                else:
                    pass
        if len(anchors) > len(best_anchors):
            best_anchors = anchors
    
    return best_anchors


def get_best_match(node_i_ego, node_j_edge, opt, graph1=None, graph2=None):
    distance = np.abs(node_i_ego[:, np.newaxis] - node_j_edge).sum(axis=2)
    error = 100
    # match = []
    match = find_anchors(distance, max_error=opt.anchor_error, graph1=graph1, graph2=graph2)
    anchors = copy.deepcopy(match)
    if len(match) < opt.min_anchors:
        return match, error
    else:
        for pair in match:
            distance[pair[0]] = 999
            distance[:,pair[1]] = 999
        for i in range(min(len(node_i_ego), len(node_j_edge)) - len(match)):
            error_item = np.min(distance)
            if error_item < opt.box_error:
                min_index = np.argmin(distance)  # Returns the index of the flattened array
                min_index_2d = np.unravel_index(min_index, distance.shape)
                if judge_box(graph1, graph2, anchors, min_index_2d, opt.box_error):
                # if np.abs(graph1[match[1][0]][min_index_2d[0]] - graph2[match[1][1]][min_index_2d[1]]) < max_error:
                    match.append(min_index_2d)
                    error += error_item
                    distance[min_index_2d[0]] = 999
                    distance[:,min_index_2d[1]] = 999
                else:
                    distance[min_index_2d[0],min_index_2d[1]] = 999
            else:
                break
        if len(match) == 0:
            match = [0]
        return match, error/np.square(len(match))

def greedy_find_matching(node_i_ego, graph_edge, i, opt, graph1=None, graph2=None):
    match_list = {}
    best_avg_err = 100
    best_matching_ever = []
    for j in range(graph_edge.shape[0]):
        node_j_edge = graph_edge[j]
        best_matching, avg_error = get_best_match(node_i_ego, node_j_edge, opt, graph1, graph2)
        # best_matching.append((i,j))
        # match_list.append({'match': best_matching, 'err': avg_error})

        if len(best_matching) >= opt.min_anchors:
            if avg_error <= best_avg_err:
                best_matching_ever = best_matching
                best_avg_err = avg_error
    return best_matching_ever, best_avg_err

def find_common_subgraph(graph1, graph2, opt):
    best_error = 100
    best_match = []
    for i in range(graph1.shape[0]):
        match, error = greedy_find_matching(graph1[i], graph2, i, opt, graph1, graph2)
        if len(match) > opt.min_anchors:
            if error <= best_error:
                best_match = match
                best_error = error
    return len(best_match), best_match
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
        noisy_t_matrix = transform[cav]
        noisy_t_matrix[0:3,-1] -= noisy_t_matrix[0:3,-1]
        # if cav != 0:
        #     noisy_t_matrix = add_rot_noise(noisy_t_matrix)
        clean_box_ego = project_box3d(box[cav], transform[cav])
        box_ego = project_box3d(box[cav], noisy_t_matrix)
        for i in range(box_ego.shape[0]):
            for j in range(box_ego.shape[0]):
                x_i = box_ego[i,:,0].mean(0)
                x_j = box_ego[j,:,0].mean(0)
                y_i = box_ego[i,:,1].mean(0)
                y_j = box_ego[j,:,1].mean(0)
                tensor[cav][i,j,0] = torch.tensor(np.sqrt(np.square(((x_i) - (x_j))) + np.square((((y_i) - (y_j))))))
    return tensor, clean_box_ego

def main(file, opt):
    # file = 'opencood/logs/opv2v_max_2023_03_29_19_10_22/single_pred/test/2021_08_20_21_10_24/000069.pt'
    single_pred = torch.load(file)

    box = copy.deepcopy(single_pred['pred']) # cav, box, 8 , 3 
    transform = single_pred['transform'].float().cpu() # 1, 5, 5, 4, 4
    if len(box) <= 1:
        return "skip"
    tensor = [np.zeros((box[0].shape[0], box[0].shape[0], 1)), np.zeros((box[1].shape[0], box[1].shape[0], 1))]

    # tensor, box[1] = coor_trans(box, tensor, transform)
    tensor, _ = coor_trans(box, tensor, transform)
    # tensor, _ = coor_trans(box, tensor, transform)
    # tensor = coor_trans_new(single_pred)
    mcs_size, matched = find_common_subgraph(tensor[0], tensor[1], opt)

    print(matched)
    print("Maximum Common Subgraph size:", mcs_size)
    if mcs_size >= 3:
        # shift_tensor = optimize(matched, box_ego = box[0], box_edge = box[1])
        # three_variable_shift_tensor = optimize_delta_theta(matched, box_ego = single_pred['pose'][0][:,0:2], box_edge_raw = single_pred['pose'][1][:,0:2], noise_rotation = transform[0][1][0])
        R_estimated, T_estimated = optimize_rt(matched, box_ego = single_pred['pred'][0].mean(1)[:,:2], box_edge = single_pred['pred'][1].mean(1)[:,:2])
        
        # error = single_pred['transform'][0,1,0,0:3,3] - shift_tensor
        # error = np.array(single_pred['transform'][0,1,0,1,0]) - four_variable_optimize_matrix
        # error = np.array([single_pred['transform'][0,1,0,0,0], single_pred['transform'][0,1,0,1,1], single_pred['transform'][0,1,0,0,3],  single_pred['transform'][0,1,0,1,3]]) - four_variable_optimize_matrix
        error_r = np.abs(R_estimated - np.array(single_pred['transform'][0,1,0,0:2,0:2]))
        error_t = np.abs(T_estimated - np.array(single_pred['transform'][0,1,0,0:2,3]))
        print('error of rotation is {}, error of shift is {}'.format(error_r, error_t))
    else:
        return "defeat"
    return error_r, error_t

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Image pair matching and pose evaluation with SuperGlue',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--data_path', type=str, default='opencood/logs/opv2v_max_2023_03_29_19_10_22/train/single_pred_pose/',
        help='Path to data.')
    
    parser.add_argument(
        '--min_anchors', type=int, default=5,
        help='Path to data.')
    
    parser.add_argument(
        '--anchor_error', type=float, default=0.3,
        help='Path to data.')
    
    parser.add_argument(
        '--box_error', type=float, default=0.5,
        help='Path to data.')
    
    opt = parser.parse_args()

    error_file_list = []
    error_r_list = []
    error_t_list = []
    # shift_tensor = []
    defeat_count = 0
    skip_count = 0
    # file_list = os.listdir('opencood/logs/opv2v_max_2023_03_29_19_10_22/single_pred/test/')
    # file_list = ["opencood/logs/dairv2x_point_pillar_lidar_max_2023_04_13_13_34_49/single_pred/test/014270_012819.pt"]
    file_list = []
    for root, dirs, files in os.walk(opt.data_path):
        for file in files:
            file_list.append(os.path.join(root, file))
    
    for file in tqdm(file_list):
        match_result = main(file, opt)
        if match_result == 'defeat':
            defeat_count += 1
        elif match_result == 'skip':
            skip_count += 1
        else:
            error_r, error_t = match_result

            if error_t.mean() > 4:
                error_file_list.append(file)
            # else:
            error_r_list.append(error_r)
            error_t_list.append(error_t)
    print(error_file_list, len(error_file_list))
    print('error of rotation is', np.abs(np.stack(error_r_list, axis=0)).mean(0))
    print('error of shift is', np.abs(np.stack(error_t_list, axis=0)).mean(0))
    # print(torch.abs(torch.stack(error_t, dim=0)).mean(0))
    print("successful rate {}".format(1-(defeat_count/(len(file_list)-skip_count))))