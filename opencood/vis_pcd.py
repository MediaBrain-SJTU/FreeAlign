import os 
import numpy as np
import torch
from matplotlib import pyplot as plt
from opencood.utils.pcd_utils import pcd_to_np, mask_ego_points
from opencood.utils.box_utils import project_box3d, project_points_by_matrix_torch
import opencood.visualization.simple_plot3d.canvas_bev as canvas_bev


def visualize(pred_box_tensor, gt_tensor, pcd, save_path):
        """
        Visualize the prediction, ground truth with point cloud together.

        Parameters
        ----------
        pred_box_tensor : torch.Tensor
            (N, 8, 3) prediction.

        gt_tensor : torch.Tensor
            (N, 8, 3) groundtruth bbx

        pcd : torch.Tensor
            PointCloud, (N, 4).

        save_path : str
            Save the visualization results to given path.
        """

        pcd_np = pcd#.cpu().numpy()
        pred_box_np = pred_box_tensor.cpu().numpy()
        gt_box_np = gt_tensor.cpu().numpy()

        plt.figure(dpi=400)
        # draw point cloud. It's in lidar coordinate
        plt.scatter(pcd_np[:,0], pcd_np[:,1], s=0.1)

        N = gt_tensor.shape[0]
        for i in range(N):
            plt.plot(gt_box_np[i,:,0], gt_box_np[i,:,1], c= "g", marker='.', linewidth=0.5, markersize=0.5)

        N = pred_box_tensor.shape[0]
        for i in range(N):
            plt.plot(pred_box_np[i,:,0], pred_box_np[i,:,1], c= "r", marker='.', linewidth=0.5, markersize=0.5)
        

        plt.savefig(save_path)
        plt.clf()
        
pcd_ego = pcd_to_np("dataset/OPV2V/test/2021_08_21_09_28_12/623/000215.pcd")
pcd_ego = mask_ego_points(pcd_ego)
pcd_ego = pcd_ego[:, :3]
pcd_edge = pcd_to_np("dataset/OPV2V/test/2021_08_21_09_28_12/632/000215.pcd")
pcd_edge = mask_ego_points(pcd_edge)
pcd_edge = pcd_edge[:, :3]
pcd_edge2 = pcd_to_np("dataset/OPV2V/test/2021_08_21_09_28_12/641/000215.pcd")
pcd_edge2 = pcd_edge2[:, :3]
pcd_edge2 = mask_ego_points(pcd_edge2)

single_pred = torch.load("opencood/logs/opv2v_max_2023_03_29_19_10_22/test/single_pred/2021_08_21_09_28_12/000215.pt")
transform = single_pred['transform'].float().cpu()
coop_pred = torch.load("opencood/logs/opv2v_max_2023_03_29_19_10_22/vis_teaser/2021_08_21_09_28_12/000215.pt")
transformation_matrix = transform[0,1,0]

project_pcd_edge = project_points_by_matrix_torch(pcd_edge, transform[0,1,0])
project_pcd_edge2 = project_points_by_matrix_torch(pcd_edge2, transform[0,2,0])

pred_box_tensor = single_pred['pred']
gt_box_tensor = coop_pred['infer_result']['gt_box_tensor'].cpu()
ego_pred_box = pred_box_tensor[0]
edge_pred_box = pred_box_tensor[1]
project_edge_pred_box = project_box3d(edge_pred_box, transformation_matrix)

pc_range = coop_pred['gt_range']
plt.figure(figsize=[(pc_range[3]-pc_range[0])/40, (pc_range[4]-pc_range[1])/40])
pc_range = [int(i) for i in pc_range]
canvas = canvas_bev.Canvas_BEV_heading_right(canvas_shape=((pc_range[4]-pc_range[1])*10, (pc_range[3]-pc_range[0])*10),
                                            canvas_x_range=(pc_range[0], pc_range[3]), 
                                            canvas_y_range=(pc_range[1], pc_range[4]),
                                            left_hand=True,
                                            canvas_bg_color=(255, 255, 255)) 

canvas_xy, valid_mask = canvas.get_canvas_coords(pcd_ego) # Get Canvas Coords
canvas.draw_canvas_points(canvas_xy[valid_mask], colors=(36, 136, 211)) # Only draw valid points
plt.axis("off")

plt.imshow(canvas.canvas)
plt.tight_layout()
plt.savefig("opencood/logs/opv2v_max_2023_03_29_19_10_22/pcd_ego.png", transparent=False, dpi=500)
plt.clf()

plt.figure(figsize=[(pc_range[3]-pc_range[0])/40, (pc_range[4]-pc_range[1])/40])
pc_range = [int(i) for i in pc_range]
canvas = canvas_bev.Canvas_BEV_heading_right(canvas_shape=((pc_range[4]-pc_range[1])*10, (pc_range[3]-pc_range[0])*10),
                                            canvas_x_range=(pc_range[0], pc_range[3]), 
                                            canvas_y_range=(pc_range[1], pc_range[4]),
                                            left_hand=True,
                                            canvas_bg_color=(255, 255, 255)) 

canvas_xy, valid_mask = canvas.get_canvas_coords(pcd_edge) # Get Canvas Coords
canvas.draw_canvas_points(canvas_xy[valid_mask], colors=(22,54,139)) # Only draw valid points
plt.axis("off")

plt.imshow(canvas.canvas)
plt.tight_layout()
plt.savefig("opencood/logs/opv2v_max_2023_03_29_19_10_22/pcd_edge.png", transparent=False, dpi=500)
plt.clf()

plt.figure(figsize=[(pc_range[3]-pc_range[0])/40, (pc_range[4]-pc_range[1])/40])
pc_range = [int(i) for i in pc_range]
canvas = canvas_bev.Canvas_BEV_heading_right(canvas_shape=((pc_range[4]-pc_range[1])*10, (pc_range[3]-pc_range[0])*10),
                                            canvas_x_range=(pc_range[0], pc_range[3]), 
                                            canvas_y_range=(pc_range[1], pc_range[4]),
                                            left_hand=True,
                                            canvas_bg_color=(255, 255, 255)) 

canvas_xy, valid_mask = canvas.get_canvas_coords(pcd_edge2) # Get Canvas Coords
canvas.draw_canvas_points(canvas_xy[valid_mask], colors=(244, 207, 204)) # Only draw valid points
plt.axis("off")

plt.imshow(canvas.canvas)
plt.tight_layout()
plt.savefig("opencood/logs/opv2v_max_2023_03_29_19_10_22/pcd_edge2.png", transparent=False, dpi=500)
plt.clf()

plt.figure(figsize=[(pc_range[3]-pc_range[0])/40, (pc_range[4]-pc_range[1])/40])
pc_range = [int(i) for i in pc_range]
canvas = canvas_bev.Canvas_BEV_heading_right(canvas_shape=((pc_range[4]-pc_range[1])*10, (pc_range[3]-pc_range[0])*10),
                                            canvas_x_range=(pc_range[0], pc_range[3]), 
                                            canvas_y_range=(pc_range[1], pc_range[4]),
                                            left_hand=True,
                                            canvas_bg_color=(255, 255, 255)) 

canvas_xy, valid_mask = canvas.get_canvas_coords(pcd_ego) # Get Canvas Coords
# canvas.draw_canvas_points(canvas_xy[valid_mask], colors=(36, 136, 211)) #149,167,126 Only draw valid points
# canvas_xy, valid_mask = canvas.get_canvas_coords(project_pcd_edge) # Get Canvas Coords
canvas.draw_canvas_points(canvas_xy[valid_mask], colors=(22,54,139)) # Only draw valid points
canvas_xy, valid_mask = canvas.get_canvas_coords(project_pcd_edge2) # Get Canvas Coords
canvas.draw_canvas_points(canvas_xy[valid_mask], colors=(209,41,25)) # Only draw valid points
plt.axis("off")

plt.imshow(canvas.canvas)
plt.tight_layout()
plt.savefig("opencood/logs/opv2v_max_2023_03_29_19_10_22/pcd_coop.png", transparent=False, dpi=500)
plt.clf()
