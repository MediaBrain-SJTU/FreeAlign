import torch
import numpy as np
import os
from opencood.utils.box_utils import project_box3d
from opencood.utils.common_utils import convert_format, compute_iou

def cal_iou_matrix(cav0_box, cav1_box, ther = 0.1):
    box0 = cav0_box[:, :4, :2]
    box1 = cav1_box[:, :4, :2]
    polygons0 = convert_format(box0)
    polygons1 = convert_format(box1)
    iou_matrix = np.zeros((box0.shape[0], box1.shape[0]))
    for i in range(box0.shape[0]):
        iou_matrix[i] = compute_iou(polygons0[i], polygons1[:])
    potential_pair = np.argmax(iou_matrix, 1)
    pair = []
    iou = []
    for i in range(len(potential_pair)):
        if iou_matrix[i, potential_pair[i]] > ther:
            pair.append((i, potential_pair[i])) 
            iou.append(iou_matrix[i, potential_pair[i]])
    return pair, iou