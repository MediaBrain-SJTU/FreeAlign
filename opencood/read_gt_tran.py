import torch
import numpy as np
from tqdm import tqdm
import os
file_list = []
root_dir = "opencood/logs/opv2v_max_2023_03_29_19_10_22/train/single_pred_pose"
for root, dirs, files in os.walk(root_dir):
    for file in files:
        file_list.append(os.path.join(root, file))

trans_list = []
for file in tqdm(file_list):
    single_pred = torch.load(file)
    trans_list.append(np.abs(single_pred['transform'][0,1,0,0:2,3]))
    
print(np.abs(np.stack(trans_list, axis=0)).mean(0))
    