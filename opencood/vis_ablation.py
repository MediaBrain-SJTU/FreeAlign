import os 
import numpy as np
import torch
from matplotlib import pyplot as plt
from opencood.utils.pcd_utils import pcd_to_np, mask_ego_points
from opencood.utils.box_utils import project_box3d, project_points_by_matrix_torch
import opencood.visualization.simple_plot3d.canvas_bev as canvas_bev


def load_pcd(path):
    pcd = pcd_to_np(path)
    pcd = mask_ego_points(pcd)
    pcd = pcd[:, :3]
    return pcd

# pcd_path_ego = "dataset/OPV2V/test/2021_08_23_16_06_26/243/000085.pcd"
# pcd_path_edge = "dataset/OPV2V/test/2021_08_23_16_06_26/252/000085.pcd"
pcd_path_ego = "dataset/OPV2V/test/2021_08_22_09_08_29/5933/000070.pcd"
pcd_path_edge = "dataset/OPV2V/test/2021_08_22_09_08_29/5942/000070.pcd"
pcd_ego = load_pcd(pcd_path_ego)
pcd_edge = load_pcd(pcd_path_edge) 


single_pred = torch.load('opencood/logs/opv2v_max_2023_03_29_19_10_22/test/single_pred/2021_08_22_09_08_29/000070.pt')
transform = single_pred['transform'].float().cpu()
coop_pred = torch.load("opencood/logs/opv2v_max_2023_03_29_19_10_22/vis_teaser/2021_08_22_09_08_29/000070.pt")
gt_box_tensor = coop_pred['infer_result']['gt_box_tensor'].cpu()
gt_box_tensor = torch.cat([gt_box_tensor[:7], gt_box_tensor[8:]], 0)
coop_pred_box_tensor = coop_pred['infer_result']['pred_box_tensor'].cpu()
transformation_matrix = transform[0,1,0]
project_pcd_edge = project_points_by_matrix_torch(pcd_edge, transform[0,1,0])

pred_box_tensor = single_pred['pred']
ego_pred_box = pred_box_tensor[0]
edge_pred_box = pred_box_tensor[1]
project_edge_pred_box = project_box3d(edge_pred_box, transformation_matrix)

ego_pose = single_pred['pose'][0][:,:2]
edge_pose = single_pred['pose'][1][:,:2]

# match_pair = [(1, 12), (9, 14), (17, 5), (6, 2), (10, 16), (4, 8), (0, 6), (2, 0)]
# match_pair = [(14, 13), (9, 1), (0, 5), (5, 6), (7, 9), (17, 7), (3, 14), (18, 0), (10, 15), (4, 12)]
match_pair = [(6, 0), (3, 2), (2, 4), (7, 1), (0, 5)]
# match_pair = [(8, 4), (5, 3), (7, 5), (6, 2), (2, 1)]
text_ego = [" "] * ego_pred_box.shape[0]
text_edge = [" "] * edge_pred_box.shape[0]

line_idx = [0,2,4,1,3]
ego_line_idx = [match_pair[i][0] for i in line_idx]
edge_line_idx = [match_pair[i][1] for i in line_idx]
ego_line_start = ego_line_idx
ego_line_end = ego_line_idx[1:] + ego_line_idx[:1]
ego_start_point = [ego_pose[i].unsqueeze(0) for i in ego_line_start]
ego_end_point = [ego_pose[i].unsqueeze(0) for i in ego_line_end]
ego_start_point = torch.cat(ego_start_point, 0)
ego_end_point = torch.cat(ego_end_point, 0)
edge_line_start = edge_line_idx
edge_line_end = edge_line_idx[1:] + edge_line_idx[:1]
edge_start_point = [edge_pose[i].unsqueeze(0) for i in edge_line_start]
edge_end_point = [edge_pose[i].unsqueeze(0) for i in edge_line_end]
edge_start_point = torch.cat(edge_start_point, 0)
edge_end_point = torch.cat(edge_end_point, 0)

# for i, pair in enumerate(match_pair):
#     text_ego[pair[0]] = str(i)
#     text_edge[pair[1]] = str(i)
    
pc_range = coop_pred['gt_range']

plt.figure(figsize=[(pc_range[3]-pc_range[0])/40, (pc_range[4]-pc_range[1])/40])
pc_range = [int(i) for i in pc_range]
canvas = canvas_bev.Canvas_BEV_heading_right(canvas_shape=((pc_range[4]-pc_range[1])*10, (pc_range[3]-pc_range[0])*10),
                                            canvas_x_range=(pc_range[0], pc_range[3]), 
                                            canvas_y_range=(pc_range[1], pc_range[4]),
                                            left_hand=True,
                                            canvas_bg_color=(0, 0, 0)) 

canvas_xy, valid_mask = canvas.get_canvas_coords(pcd_ego) # Get Canvas Coords
canvas.draw_canvas_points(canvas_xy[valid_mask], colors=(251,210,62)) # Only draw valid points
canvas.draw_lines(ego_start_point, ego_end_point, colors=(255,255,255),box_line_thickness=4)
canvas.draw_boxes(ego_pred_box,colors=(255,0,0),texts=text_ego,box_line_thickness=4,box_text_size=1,) 
plt.axis("off")
plt.imshow(canvas.canvas)
plt.tight_layout()
plt.savefig("opencood/logs/opv2v_max_2023_03_29_19_10_22/ablation_ego.png", transparent=False, dpi=500)
plt.clf()

plt.figure(figsize=[(pc_range[3]-pc_range[0])/40, (pc_range[4]-pc_range[1])/40])
pc_range = [int(i) for i in pc_range]
canvas = canvas_bev.Canvas_BEV_heading_right(canvas_shape=((pc_range[4]-pc_range[1])*10, (pc_range[3]-pc_range[0])*10),
                                            canvas_x_range=(pc_range[0], pc_range[3]), 
                                            canvas_y_range=(pc_range[1], pc_range[4]),
                                            left_hand=True,
                                            canvas_bg_color=(0, 0, 0)) 

canvas_xy, valid_mask = canvas.get_canvas_coords(pcd_edge) # Get Canvas Coords
canvas.draw_canvas_points(canvas_xy[valid_mask], colors=(66,196,201)) # Only draw valid points
canvas.draw_lines(edge_start_point, edge_end_point, colors=(255,255,255),box_line_thickness=4)
canvas.draw_boxes(edge_pred_box,colors=(255,0,0),texts=text_edge,box_line_thickness=4,box_text_size=1) 
plt.axis("off")
plt.imshow(canvas.canvas)
plt.tight_layout()
plt.savefig("opencood/logs/opv2v_max_2023_03_29_19_10_22/ablation_edge.png", transparent=False, dpi=500)
plt.clf()

plt.figure(figsize=[(pc_range[3]-pc_range[0])/40, (pc_range[4]-pc_range[1])/40])
pc_range = [int(i) for i in pc_range]
canvas = canvas_bev.Canvas_BEV_heading_right(canvas_shape=((pc_range[4]-pc_range[1])*10, (pc_range[3]-pc_range[0])*10),
                                            canvas_x_range=(pc_range[0], pc_range[3]), 
                                            canvas_y_range=(pc_range[1], pc_range[4]),
                                            left_hand=True,
                                            canvas_bg_color=(0, 0, 0)) 

canvas_xy, valid_mask = canvas.get_canvas_coords(pcd_ego) # Get Canvas Coords
canvas.draw_canvas_points(canvas_xy[valid_mask], colors=(251,210,62)) # Only draw valid points
canvas_xy, valid_mask = canvas.get_canvas_coords(project_pcd_edge) # Get Canvas Coords
canvas.draw_canvas_points(canvas_xy[valid_mask], colors=(66,196,201)) # Only draw valid points
canvas.draw_lines(ego_start_point, ego_end_point, colors=(255,255,255),box_line_thickness=4)
canvas.draw_boxes(gt_box_tensor,colors=(0,255,0),box_line_thickness=4) 
canvas.draw_boxes(coop_pred_box_tensor,colors=(255,0,0),box_line_thickness=4)


plt.axis("off")
plt.imshow(canvas.canvas)
plt.tight_layout()
plt.savefig("opencood/logs/opv2v_max_2023_03_29_19_10_22/abalation_gt.png", transparent=False, dpi=500)
plt.clf()