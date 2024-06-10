import numpy as np
import torch
from opencood.utils.box_utils import project_box3d
from opencood.utils.pose_utils import generate_noise
from opencood.utils.transformation_utils import tfm_to_pose, pose_to_tfm
from freealign.optimize import optimize, optimize_rt, optimize_delta_theta, optimize_rt_raw
import time
import os
import copy
import cv2
from tqdm import tqdm
import argparse

def add_rot_noise(clean_tfm):
    pose = tfm_to_pose(clean_tfm)
    pose += generate_noise(0, 0.0)
    tfm = pose_to_tfm(np.expand_dims(pose, 0))
    return tfm.squeeze()

def find_closest_points(src_pts, dst_pts):
    closest_points = []
    for pt in src_pts:
        diff = dst_pts - pt
        distances = np.linalg.norm(diff, axis=1)
        closest_points.append(dst_pts[np.argmin(distances)])
    return np.stack(closest_points, axis=0)

def estimate_transform(src_pts, dst_pts):
    src_mean = np.mean(src_pts, axis=0)
    dst_mean = np.mean(dst_pts, axis=0)

    src_centered = src_pts - src_mean
    dst_centered = dst_pts - dst_mean

    H = np.dot(src_centered.T, dst_centered)
    U, _, Vt = np.linalg.svd(H)

    rotation = np.dot(Vt.T, U.T)
    translation = dst_mean - np.dot(src_mean, rotation.T)

    return rotation, translation

def icp(src_pts, dst_pts, max_iterations, tolerance=1e-6):
    prev_error = np.inf
    rotation = np.eye(2)
    translation = np.zeros(2)

    for _ in range(max_iterations):
        # Transform the source points
        src_transformed = np.dot(src_pts, rotation.T) + translation

        # Find the closest points in the destination set
        closest_dst_pts = find_closest_points(src_transformed, dst_pts)

        # Estimate the transformation
        R, T = estimate_transform(src_transformed, closest_dst_pts)

        # Update the overall transformation
        rotation = np.dot(R, rotation)
        translation = np.dot(R, translation) + T

        # Compute the error
        error = np.mean(np.linalg.norm(src_transformed - closest_dst_pts, axis=1))

        # Check for convergence
        # if np.abs(prev_error - error) < tolerance:
        #     break

        prev_error = error

    return rotation, translation


def main(file, opt):
    # file = 'opencood/logs/opv2v_max_2023_03_29_19_10_22/single_pred/test/2021_08_20_21_10_24/000069.pt'
    single_pred = torch.load(file)

    box = copy.deepcopy(single_pred['pred']) # cav, box, 8 , 3 
    transform = single_pred['transform'].float().cpu() # 1, 5, 5, 4, 4
    if len(box) <= 1:
        return "skip"
    # tensor = [np.zeros((box[0].shape[0], box[0].shape[0], 1)), np.zeros((box[1].shape[0], box[1].shape[0], 1))]
    box_ego = box[0][:,:,0:2].mean(1)
    box_edge = box[1][:,:,0:2].mean(1)
    R_estimated, T_estimated = icp(box_ego, box_edge, max_iterations=1000)
    error_r = np.abs(R_estimated - np.array(single_pred['transform'][0,1,0,0:2,0:2]))
    error_t = np.abs(T_estimated - np.array(single_pred['transform'][0,1,0,0:2,3]))
    print('error of rotation is {}, error of shift is {}'.format(error_r, error_t))


    return error_r, error_t
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Image pair matching and pose evaluation with SuperGlue',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--data_path', type=str, default='opencood/logs/opv2v_max_2023_03_29_19_10_22/test/single_pred_pose/',
        help='Path to data.')
    
    parser.add_argument(
        '--min_anchors', type=int, default=3,
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
    time1 = time.time()
    count = 0
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

            if error_t.mean() > 3:
                error_file_list.append(file)
            # else:
            error_r_list.append(error_r)
            error_t_list.append(error_t)
            count += 1
        print('average time is {}'.format((time.time()-time1)/count))
    print(error_file_list, len(error_file_list))
    print('error of rotation is', np.abs(np.stack(error_r_list, axis=0)).mean(0))
    print('error of shift is', np.abs(np.stack(error_t_list, axis=0)).mean(0))
    # print(torch.abs(torch.stack(error_t, dim=0)).mean(0))
    print("successful rate {}".format(1-(defeat_count/(len(file_list)-skip_count))))