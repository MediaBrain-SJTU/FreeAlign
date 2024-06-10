import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from opencood.utils.box_utils import project_box3d
import cv2
import open3d as o3d

def to_point_cloud(points):
    points_3d = np.hstack((points, np.zeros((len(points), 1))))
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_3d)
    return point_cloud


# def icp(match_list, box_ego, box_edge, max_iterations=1000, tolerance=1e-6, rotation = np.eye(2), translation = np.zeros(2)):
#     dst_pts = np.array([np.array(box_edge[match_list[i][1]][0:2]) for i in range(len(match_list))])
#     src_pts = np.array([np.array( box_ego[match_list[i][0]][0:2]) for i in range(len(match_list))])
#     src_pts = np.dot(src_pts, rotation.T) + translation
#     assert dst_pts.shape == src_pts.shape, "The input point sets should have the same shape"
#     assert dst_pts.shape[1] == 2, "The input point sets should have 2D coordinates"
def icp(src_pts, dst_pts, max_iterations, tolerance=1e-1):
    src_cloud = to_point_cloud(src_pts)
    dst_cloud = to_point_cloud(dst_pts)

    transformation_init = np.eye(4)
    icp_result = o3d.pipelines.registration.registration_icp(
        src_cloud, dst_cloud, tolerance, transformation_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations)
    )

    rotation = icp_result.transformation[:2, :2]
    translation = icp_result.transformation[:2, 3]

    return rotation, translation



def find_transform_process(match_list, box_ego, box_edge):
    points_A = np.array([np.array(box_edge[match_list[i][1]][0:2]) for i in range(len(match_list))])
    points_B = np.array([np.array( box_ego[match_list[i][0]][0:2]) for i in range(len(match_list))])
    assert points_A.shape == points_B.shape, "The input point sets should have the same shape"
    assert points_A.shape[1] == 2, "The input point sets should have 2D coordinates"
    
    # Find centroids of each point set
    centroid_A = np.mean(points_A, axis=0)
    centroid_B = np.mean(points_B, axis=0)

    # Center the point sets
    centered_A = points_A - centroid_A
    centered_B = points_B - centroid_B

    # Calculate the covariance matrix
    covariance_matrix = centered_A.T @ centered_B

    # Perform Singular Value Decomposition
    U, _, Vt = np.linalg.svd(covariance_matrix)

    # Calculate the rotation matrix
    R = Vt.T @ U.T

    # Ensure proper rotation (prevent reflections)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Calculate the shift vector
    t = centroid_B - R @ centroid_A

    return R, t



def find_transform(points_A, points_B):
    assert points_A.shape == points_B.shape, "The input point sets should have the same shape"
    assert points_A.shape[1] == 2, "The input point sets should have 2D coordinates"
    
    # Find centroids of each point set
    centroid_A = np.mean(points_A, axis=0)
    centroid_B = np.mean(points_B, axis=0)

    # Center the point sets
    centered_A = points_A - centroid_A
    centered_B = points_B - centroid_B

    # Calculate the covariance matrix
    covariance_matrix = centered_A.T @ centered_B

    # Perform Singular Value Decomposition
    U, _, Vt = np.linalg.svd(covariance_matrix)

    # Calculate the rotation matrix
    R = Vt.T @ U.T

    # Ensure proper rotation (prevent reflections)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Calculate the shift vector
    t = centroid_B - R @ centroid_A

    return R, t

def optimize_rt_raw(match_list, box_ego, box_edge):
    points_A = np.array([np.array(box_edge[match_list[i][1]][0:2]) for i in range(len(match_list))])
    points_B = np.array([np.array( box_ego[match_list[i][0]][0:2]) for i in range(len(match_list))])
    M, _ = cv2.estimateAffinePartial2D(points_A.reshape((-1,points_A.shape[-1])), points_B.reshape((-1,points_A.shape[-1])), method=cv2.LMEDS)
    # R, t = find_transform(points_A, points_B)
    # return [R[0,0], R[1,0], t[0], t[1]]
    # matrix = np.identity(4)
    # matrix[0:3,0:3] = R
    # matrix[3,0:3] = t
    # return R, t
    return M[:,0:2], M[:,2]

def optimize_rt(match_list, box_ego, box_edge):
    points_A = np.array([np.array(box_edge[match_list[i][1]].mean(0)[0:2]) for i in range(len(match_list))])
    points_B = np.array([np.array( box_ego[match_list[i][0]].mean(0)[0:2]) for i in range(len(match_list))])
    M, _ = cv2.estimateAffinePartial2D(points_A.reshape((-1,points_A.shape[-1])), points_B.reshape((-1,points_A.shape[-1])))
    # R, t = find_transform(points_A, points_B)
    # return [R[0,0], R[1,0], t[0], t[1]]
    # matrix = np.identity(4)
    # matrix[0:3,0:3] = R
    # matrix[3,0:3] = t
    # return R, t
    return M[:,0:2], M[:,2]

def optimize_rt_3D(match_list, box_ego, box_edge):
    points_A = np.array([np.array(box_edge[match_list[i][1]]) for i in range(len(match_list))])
    points_B = np.array([np.array( box_ego[match_list[i][0]]) for i in range(len(match_list))])
    M= cv2.estimateAffine3D(points_A, points_B)
    # R, t = find_transform(points_A, points_B)
    # return [R[0,0], R[1,0], t[0], t[1]]
    # matrix = np.identity(4)
    # matrix[0:3,0:3] = R
    # matrix[3,0:3] = t
    # return R, t
    return M[1][0:2,0:2], M[1][0:2,3]

# def optimize_rt(match_list, box_ego, box_edge):
#     x = np.zeros((2*len(match_list)))
#     b = np.zeros((2*len(match_list)))
#     A = np.zeros((2*len(match_list), 4))
#     for i in range(len(match_list)):
#         x[i*2:(i+1)*2] = box_edge[match_list[i][1]]
#         b[i*2:(i+1)*2] = box_ego[match_list[i][0]]
#         A[i*2,0] = x[i*2]
#         A[i*2,1] = -x[i*2+1]
#         A[i*2,2] = 1
#         A[i*2,3] = 0
#         A[i*2+1,0] = x[i*2+1]
#         A[i*2+1,1] = x[i*2]
#         A[i*2+1,2] = 0
#         A[i*2+1,3] = 1
#     para = np.matmul(np.matmul(np.linalg.inv(np.matmul(A.T, A)),A.T),b)
#     return para


def optimize(match_list, box_ego, box_edge):
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=4)
    # error = torch.Tensor(1,3)

    # for i in range(len(file_list)):
    # for file in match_list:
    #     # match_list, box_ego, box_edge = torch.load(file)
    shift_tensor = get_shift_tensor(match_list, box_ego, box_edge)
    print('Average shift_tensor is {}'.format(shift_tensor))
    return shift_tensor

def get_shift_tensor(match_list, box_ego, box_edge):
    shift = []
    for match in match_list:
        if "None" not in match:
            [i, j] = match
            shift_item = box_ego[i].mean(0) - box_edge[j].mean(0)
            shift.append(shift_item)
            # print("shift between {} {} is {}".format(i,j,shift_item))

    shift_tensor = torch.stack(shift)
    shift_tensor = torch.mean(shift_tensor, dim = 0)
    return shift_tensor

def optimize_delta_theta(match_list, box_ego, box_edge_raw, noise_rotation) -> list:
    # noise_yaw = [noise_rotation[0,0], noise_rotation[1,0]]
    b, A = establish_linear_equations(match_list, box_ego, box_edge_raw, noise_rotation[0:2,0:2])
    para = np.matmul(np.matmul(np.linalg.inv(np.matmul(A.T, A)),A.T),b)
    return para

def establish_linear_equations(match_list, box_ego, box_edge_raw, noise_rotation):
    b = np.zeros((2*len(match_list)))
    A = np.zeros((2*len(match_list), 3))
    for i in range(len(match_list)):
        A[i*2:(i+1)*2], b[i*2:(i+1)*2] = a_matched_equations(match_list[i], box_ego[match_list[i][0]], box_edge_raw[match_list[i][1]], noise_rotation)
    return b, A

def a_matched_equations(match, box_ego_item, box_edge_raw_item, noise_rotation):
    b = np.zeros(2)
    box_rotation = np.matmul(noise_rotation, box_edge_raw_item)
    A = np.zeros((2, 3))
    k1 = -(noise_rotation[0,1] * box_edge_raw_item[0] + noise_rotation[0,0] * box_edge_raw_item[1])
    k2 = -noise_rotation[0,1] * box_edge_raw_item[1] + noise_rotation[0,0] * box_edge_raw_item[0]
    b[0] = box_edge_raw_item[0]
    b[1] = box_edge_raw_item[1]
    A[0,0] = k1
    A[0,1] = 1
    A[0,2] = 0
    A[1,0] = k2
    A[1,1] = 0
    A[1,2] = 1
    return A, b