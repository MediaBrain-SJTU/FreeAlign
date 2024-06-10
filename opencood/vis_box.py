import os 
import numpy as np
import torch
from matplotlib import pyplot as plt
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

    pred_box_np = pred_box_tensor.cpu().numpy()
    gt_box_np = gt_tensor.cpu().numpy()

    plt.figure(dpi=400)
    # draw point cloud. It's in lidar coordinate

    N = gt_tensor.shape[0]
    for i in range(N):
        plt.plot(gt_box_np[i,:,0], gt_box_np[i,:,1], c= "g", marker='.', linewidth=1.5, markersize=1.5)

    N = pred_box_tensor.shape[0]
    for i in range(N):
        plt.plot(pred_box_np[i,:,0], pred_box_np[i,:,1], c= "r", marker='.', linewidth=1.5, markersize=1.5)
    

    plt.savefig(save_path)
    plt.clf()
    
    
def draw_box(pc_range, box, color, save_path):
    box_self = torch.tensor([[[  3.0000,  -0.9604,  -1.9316],
         [  2.9570,   1.1709,  -1.9316],
         [ -1.9355,   1.0703,  -1.9316],
         [ -1.8916,  -1.0615,  -1.9316],
         [  3.0000,  -0.9604,  -0.4050],
         [  2.9570,   1.1709,  -0.4050],
         [ -1.9355,   1.0703,  -0.4050],
         [ -1.8916,  -1.0615,  -0.4050]]])
    plt.figure(figsize=[(pc_range[3]-pc_range[0])/40, (pc_range[4]-pc_range[1])/40])
    pc_range = [int(i) for i in pc_range]
    canvas = canvas_bev.Canvas_BEV_heading_right(canvas_shape=((pc_range[4]-pc_range[1])*10, (pc_range[3]-pc_range[0])*10),
                                                canvas_x_range=(pc_range[0], pc_range[3]), 
                                                canvas_y_range=(pc_range[1], pc_range[4]),
                                                left_hand=True,
                                                canvas_bg_color=(255, 255, 255)) 
    canvas.draw_boxes(box,colors=color,box_line_thickness=8)
    # canvas.draw_boxes(box_self,colors=(0,0,0))
    plt.axis("off")

    plt.imshow(canvas.canvas)
    plt.tight_layout()
    plt.savefig(save_path, transparent=False, dpi=500)
    plt.clf()
    plt.close()
    
single_pred = torch.load("opencood/logs/opv2v_max_2023_03_29_19_10_22/test/single_pred/2021_08_21_09_28_12/000215.pt")
transform = single_pred['transform'].float().cpu()
coop_pred = torch.load("opencood/logs/opv2v_max_2023_03_29_19_10_22/vis_teaser/2021_08_21_09_28_12/000215.pt")
# transformation_matrix = transform[0,1,0]

# box = single_pred['pred'][0] # cav, box, 8 , 3 
# transform = single_pred['transform'] # 1, 5, 5, 4, 4

# box_edge_project = project_box3d(single_pred['pred'][1], transform[0,1,0].float().cpu())
    
pred_box_tensor = single_pred['pred']

gt_box_tensor = coop_pred['infer_result']['gt_box_tensor'].cpu()
ego_pred_box = pred_box_tensor[0]
edge_pred_box = pred_box_tensor[1]
edge_pred_box2 = pred_box_tensor[2]
project_edge_pred_box = project_box3d(edge_pred_box, transform[0,1,0])
project_edge_pred_box2 = project_box3d(edge_pred_box2, transform[0,2,0])


pc_range = coop_pred['gt_range']
draw_box(pc_range, gt_box_tensor, (209,41,25), "opencood/logs/opv2v_max_2023_03_29_19_10_22/box_gt.png")
# draw_box(pc_range, ego_pred_box, (36,136,211), "opencood/logs/opv2v_max_2023_03_29_19_10_22/box_ego.png")
# draw_box(pc_range, edge_pred_box, (22,54,139), "opencood/logs/opv2v_max_2023_03_29_19_10_22/box_edge1.png")
# draw_box(pc_range, edge_pred_box2, (209,41,25), "opencood/logs/opv2v_max_2023_03_29_19_10_22/box_edge2.png")
# draw_box(pc_range, edge_pred_box2, (250,128,114), "opencood/logs/opv2v_max_2023_03_29_19_10_22/box_edge2_past2.png")

# plt.figure(figsize=[(pc_range[3]-pc_range[0])/40, (pc_range[4]-pc_range[1])/40])
# pc_range = [int(i) for i in pc_range]
# canvas = canvas_bev.Canvas_BEV_heading_right(canvas_shape=((pc_range[4]-pc_range[1])*10, (pc_range[3]-pc_range[0])*10),
#                                             canvas_x_range=(pc_range[0], pc_range[3]), 
#                                             canvas_y_range=(pc_range[1], pc_range[4]),
#                                             left_hand=True,
#                                             canvas_bg_color=(255, 255, 255)) 

# canvas.draw_boxes(ego_pred_box,colors=(0,129,0))
# canvas.draw_boxes(project_edge_pred_box,colors=(135,206,235))
# canvas.draw_boxes(project_edge_pred_box2,colors=(250,128,114))

# plt.axis("off")

# plt.imshow(canvas.canvas)
# plt.tight_layout()
# plt.savefig("vis_miss.png", transparent=False, dpi=500)
# plt.clf()
# plt.close()
