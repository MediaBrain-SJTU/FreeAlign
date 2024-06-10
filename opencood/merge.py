import os
import torch
import torch
import numpy as np
import os
from opencood.utils.box_utils import project_box3d
from opencood.utils.common_utils import convert_format, compute_iou

def cal_iou_matrix(cav0_box, cav1_box, ther = 0.5):
    box0 = cav0_box[:, :4, :2]
    box1 = cav1_box[:, :4, :2]
    polygons0 = convert_format(box0)
    polygons1 = convert_format(box1)
    iou_matrix = np.zeros((box0.shape[0], box1.shape[0]))
    for i in range(box0.shape[0]):
        iou_matrix[i] = compute_iou(polygons0[i], polygons1[:])
    print(box0.shape[0], box1.shape[0])
    potential_pair = np.argmax(iou_matrix, 1)
    pair = []
    iou = []
    for i in range(len(potential_pair)):
        if iou_matrix[i, potential_pair[i]] > ther:
            pair.append((i, potential_pair[i])) 
            iou.append(iou_matrix[i, potential_pair[i]])
    return pair, iou

file_list = []
root_dir = "opencood/logs/v2xset_max_2023_04_20_03_42_57/train/single_pred_pose"
for root, dirs, files in os.walk(root_dir):
    for file in files:
        file_list.append(os.path.join(root, file))
# for scenario_folder in scenario_folders:
#     print(scenario_folder)
#     # time_list = sorted([x for x in os.listdir(os.path.join(single_pred_dir, scenario_folder))])
#     time_list = sorted([x for x in os.listdir("opencood/logs/v2xsim_max_2023_04_18_05_13_50/test/single_pred_pose")])
for file in file_list:
    single_pred = torch.load(file)
    # pose = torch.load(os.path.join(root_dir, "pose", scenario_folder, timestep))
    # print(os.path.join(root_dir, "single_pred", scenario_folder, timestep))
    # print(single_pred.keys())
    # import time
    # time.sleep(1000)
    # single_pred.update({"lidar_pose": pose})
    # save_path = os.path.join(root_dir, "single_pred_pose", scenario_folder)
    # if not os.path.exists(save_path):
    #     try:
    #         os.makedirs(save_path)
    #     except FileExistsError:
    #         pass
    # torch.save(single_pred, os.path.join(save_path, timestep))
    # box_cav1 = single_pred['pred'][cav1] # cav, box, 8 , 3 
    transform = single_pred['transform'] # 1, 5, 5, 4, 4
    gt_pair = []
    print(file, len(single_pred['pred']))
    for cav1 in range(len(single_pred['pred'])):
        box_cav1_ego = project_box3d(single_pred['pred'][cav1], transform[0,cav1,0].float().cpu())
        gt_pair_cav1 = []
        for cav2 in range(len(single_pred['pred'])):
            box_cav2_ego = project_box3d(single_pred['pred'][cav2], transform[0,cav2,0].float().cpu())
            if box_cav1_ego.shape[0] == 0 or box_cav2_ego.shape[0] == 0:
                gt_pair_cav1.append([])
            else:
                pair, iou = cal_iou_matrix(box_cav1_ego, box_cav2_ego)
                gt_pair_cav1.append(pair)
        gt_pair.append(gt_pair_cav1)
    # print(gt_pair[0][0])
    # import time
    # time.sleep(1000)
    single_pred.update({"match_pair_all": gt_pair})
    torch.save(single_pred, file)

