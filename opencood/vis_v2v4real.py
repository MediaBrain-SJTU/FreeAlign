import os
from opencood.utils.box_utils import create_bbx
from opencood.utils.transformation_utils import pose_to_tfm
from matplotlib import pyplot as plt
import numpy as np
from opencood.hypes_yaml.yaml_utils import load_yaml
import opencood.visualization.simple_plot3d.canvas_bev as canvas_bev
from opencood.utils.pcd_utils import read_pcd
import opencood.utils.pcd_utils as pcd_utils
from opencood.utils.pcd_utils import pcd_to_np, mask_ego_points
from opencood.utils.box_utils import project_box3d, project_points_by_matrix_torch

def euler_to_rotation_matrix(rotation):
    roll,yaw,pitch  = rotation

    # Calculate the rotation matrix
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll), np.cos(roll)]])

    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])

    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw), np.cos(yaw), 0],
                   [0, 0, 1]])

    # Combine the individual rotation matrices
    R = Rz @ Ry @ Rx

    return R

def pose_to_transform_matrix(pose):
    # Extract translation and rotation components from the pose
    translation = pose[:3]  # x, y, z
    rotation = pose[3:]     # roll, pitch, yaw
    rotation = rotation/180 * np.pi

    # Create rotation matrix from Euler angles (roll, pitch, yaw)
    R = euler_to_rotation_matrix(rotation)

    # Create the 4x4 transformation matrix
    transform = np.eye(4)
    transform[:3, :3] = R
    transform[:3, 3] = translation

    return transform

def get_box_coord(object_content):
        
    location = object_content['location']
    rotation = object_content['angle']
    center = object_content['center']
    extent = object_content['extent']

    object_pose = [location[0] + center[0],
                    location[1] + center[1],
                    location[2] + center[2],
                    rotation[0], rotation[1], rotation[2]]
    trans_matrix = pose_to_transform_matrix(np.array(object_pose))
    bbx = create_bbx(extent).T
    bbx = np.matmul(trans_matrix[:3,:3], bbx) + np.expand_dims(trans_matrix[:3,3], 1)

    return bbx.T


def get_bbxs(object_detected):
    all_boxes = []
    for object_key in object_detected.keys():
        object_content = object_detected[object_key]
        all_boxes.append(np.expand_dims(get_box_coord(object_content), 0))
    all_boxes = np.concatenate(all_boxes)
    return all_boxes


root_dir = "dataset/v2v4real/train"
for scene in os.listdir(root_dir):
    scene_folder = os.path.join(root_dir, scene)
    timesteps = sorted(x.replace(".yaml","") for x in os.listdir(os.path.join(scene_folder,"0")) if x.endswith('.yaml'))
    for timestep in timesteps:
        ego_path = os.path.join(scene_folder, "0")
        edge_path = os.path.join(scene_folder, "1")
        
        cav1_yaml = load_yaml(os.path.join(ego_path, timestep+".yaml"))
        lidar_pose1 = cav1_yaml['lidar_pose']
        object_bbxs1 = get_bbxs(cav1_yaml["vehicles"])
        pcd_ego = pcd_utils.pcd_to_np(os.path.join(ego_path, timestep+".pcd"))
        pcd_ego = mask_ego_points(pcd_ego)
        pcd_ego = pcd_ego[:, :3]

        cav2_yaml = load_yaml(os.path.join(edge_path, timestep+".yaml"))
        lidar_pose2 = cav2_yaml['lidar_pose']
        object_bbxs2 = get_bbxs(cav2_yaml["vehicles"])
        pcd_edge = pcd_utils.pcd_to_np(os.path.join(edge_path, timestep+".pcd"))
        pcd_edge = mask_ego_points(pcd_edge)
        pcd_edge = pcd_edge[:, :3]


        trans_matrix = np.matmul(np.linalg.inv(lidar_pose1), lidar_pose2)
        project_pcd_edge = project_points_by_matrix_torch(pcd_edge, trans_matrix)

        projected_box_edge = project_box3d(object_bbxs2, trans_matrix)

        pc_range = [-140, -60, -3.5, 140, 60, 1.5]
        plt.figure(figsize=[(pc_range[3]-pc_range[0])/40, (pc_range[4]-pc_range[1])/40])
        pc_range = [int(i) for i in pc_range]
        canvas = canvas_bev.Canvas_BEV_heading_right(canvas_shape=((pc_range[4]-pc_range[1])*10, (pc_range[3]-pc_range[0])*10),
                                                    canvas_x_range=(pc_range[0], pc_range[3]), 
                                                    canvas_y_range=(pc_range[1], pc_range[4]),
                                                    left_hand=True,
                                                    canvas_bg_color=(255, 255, 255)) 

        canvas_xy, valid_mask = canvas.get_canvas_coords(pcd_ego) # Get Canvas Coords
        canvas.draw_canvas_points(canvas_xy[valid_mask], colors=(209, 41, 25)) #149,167,126 Only draw valid points
        canvas_xy, valid_mask = canvas.get_canvas_coords(project_pcd_edge) # Get Canvas Coords
        canvas.draw_canvas_points(canvas_xy[valid_mask], colors=(54,162,234)) # Only draw valid points
        canvas.draw_boxes(object_bbxs1, colors=(209, 41, 25), box_text_size=1, box_line_thickness=2)
        canvas.draw_boxes(projected_box_edge, colors=(54,162,234), box_text_size=1, box_line_thickness=2)

        # canvas_xy, valid_mask = canvas.get_canvas_coords(project_pcd_edge2) # Get Canvas Coords
        # canvas.draw_canvas_points(canvas_xy[valid_mask], colors=(240,207,204)) # Only draw valid points
        plt.axis("off")

        plt.imshow(canvas.canvas)
        plt.tight_layout()
        if not os.path.exists(os.path.join("visualize",scene)):
            os.makedirs(os.path.join("visualize",scene))
        plt.savefig(os.path.join("visualize",scene,timestep+".png"), transparent=False, dpi=500)
        plt.clf()
        plt.close()