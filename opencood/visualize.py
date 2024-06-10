import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from freealign.match.match_v7_debug import *
from opencood.utils.box_utils import project_box3d


def draw_box(box, color):
    box[:, :, 0] = (box[:, :, 0] + 100.8) / 0.8
    box[:, :, 1] = (box[:, :, 1] + 40.8) / 0.8
    for i in range(box.shape[0]):
        plt.plot([box[i,0,0],box[i,1,0]], [box[i,0,1],box[i,1,1]], color=color, linewidth=0.5)
        plt.plot([box[i,1,0],box[i,2,0]], [box[i,1,1],box[i,2,1]], color=color, linewidth=0.5)
        plt.plot([box[i,2,0],box[i,3,0]], [box[i,2,1],box[i,3,1]], color=color, linewidth=0.5)
        plt.plot([box[i,3,0],box[i,0,0]], [box[i,3,1],box[i,0,1]], color=color, linewidth=0.5)
        
        plt.text(box[i,:,0].mean(), box[i,:,1].mean()-6, i, fontsize=5, c=color)

fig, ax = plt.subplots()

# Plot the heatmap using imshow
data = np.zeros((100,352))
im = ax.imshow(data, cmap='viridis')
# failure cases ['opencood/logs/dairv2x_point_pillar_lidar_max_2023_04_13_13_34_49/train/single_pred_pose/34/020354.pt', 'opencood/logs/dairv2x_point_pillar_lidar_max_2023_04_13_13_34_49/train/single_pred_pose/34/020343.pt', 'opencood/logs/dairv2x_point_pillar_lidar_max_2023_04_13_13_34_49/train/single_pred_pose/71/003509.pt']
single_pred = torch.load('opencood/logs/opv2v_max_2023_03_29_19_10_22/test/single_pred_pose/2021_08_23_16_06_26/000333.pt')

# box = copy.deepcopy(single_pred['pred']) # cav, box, 8 , 3 
# transform = single_pred['transform'].float().cpu() # 1, 5, 5, 4, 4
# box_1_right = get_right_box(box, single_pred['gt_box_tensor'].cpu(), transform)
# single_pred['pred'][1] = box_1_right


box = single_pred['pred'][0] # cav, box, 8 , 3 
# print(box)
transform = single_pred['transform'] # 1, 5, 5, 4, 4
box = project_box3d(box, transform[0,0,0].float().cpu())
draw_box(box, 'red') #ego

box = single_pred['pred'][1]
box = project_box3d(box, transform[0,1,0].float().cpu())
draw_box(box, 'yellow') #agent

# box = single_pred['gt_box_tensor'].cpu() # gt boxes only generated for dairv2x dataset
# draw_box(box, 'green') #green
    
plt.savefig(f'matchdairg.png', dpi=300)
plt.clf()
