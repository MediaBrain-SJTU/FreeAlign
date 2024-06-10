from copy import deepcopy
from pathlib import Path
from typing import List, Tuple

import torch
from torch import nn
from concurrent import futures
from .superbevglue import cat_list
import torch.multiprocessing as mp

class SuperBox(nn.Module):
    # Extract boxes features from feature maps
    def __init__(self, config={}):
        super().__init__()
        self.in_channels = config.get('init_channels')
        # self.out_channels = config.get('init_channels', 256)
        self.out_channels = 256
        c1, c2, c3,  = 128, 128, 256
        self.conv1a = nn.Conv2d(self.in_channels, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        # self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)

        self.convDa = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(
            c3, self.out_channels,
            kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.finalconv = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1, stride=1, padding=0)
        # self.score_linear = nn.Linear(self.out_channels, 1)

    def feature_conv(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.relu(self.convDa(x))
        x = self.relu(self.convDb(x))
        return x

    def extract_boxes_descriptors(self, features_matching: torch.Tensor, boxes: list, agent_num_list: list, box_num: list) -> list:
        # input feature_matching : BA*C*H*W
        # input boxes: [[[Agent1 boxes], [Agent2 boxes], [Agent3 boxes]...], [Batch2]...]
        # output box_descriptors [[[Agent1 box descriptor], [Agent2 box descriptor], [Agent3 box descriptor]...], [Batch2]...]
        bs = int(len(agent_num_list))
        # load_list = [{'features': features_matching[5*batch: 5*batch + agent_num_list[batch]], 'boxes': boxes[batch]} for batch in range(bs)] 
        
        # pool = futures.ProcessPoolExecutor(max_workers=bs)
        # box_descriptors = list(pool.map(self.extract_descriptors_per_batch, load_list))
        # box_descriptors = []
        # for batch in range(bs):
        # box_descriptors = [self.extract_descriptors_per_batch({'features': features_matching[5*batch: 5*batch + agent_num_list[batch]], 'boxes': boxes[batch]}) for batch in range(bs)] 

        box_descriptors = []
        count = 0
        for batch in range(bs):
            box_descriptors.append(self.extract_descriptors_per_batch({'features': features_matching[count: count + agent_num_list[batch]], 'boxes': boxes[batch]}))
            count += agent_num_list[batch]
    
        return box_descriptors

    def extract_descriptors_per_batch(self, data_dict):
        boxes, features = data_dict['boxes'], data_dict['features']
        box_descriptors_per_batch = [self.extract_descriptors_per_agent({'features': features[agent], 'boxes': boxes[agent]}) for agent in range(len(boxes))] 
        return box_descriptors_per_batch

    def extract_descriptors_per_agent(self, data_dict):
        box_descriptors_per_agent = []
        boxes, feature = data_dict['boxes'], data_dict['features']
        c, h, w = feature.shape
        box_descriptors_per_agent = torch.stack([self.get_box_feature(box, feature) for box in boxes], dim=0)
        return box_descriptors_per_agent
    
    def get_box_feature(self, box, feature):
        c, h, w = feature.shape
        # left, right, up, down = box[:,0].min(), box[:,0].max(), box[:,1].min(), box[:,1].max()
        # left_bev, right_bev, up_bev, down_bev = self.cartesian2BEV([left, right, up, down], h, w)
        left_bev, right_bev, up_bev, down_bev = box.long()
        assert right_bev > left_bev
        assert down_bev > up_bev
        box_feature = feature[:, up_bev:down_bev, left_bev:right_bev]
        box_feature = self.relu(self.finalconv(box_feature.unsqueeze(0)))
        box_feature = self.pool(box_feature).squeeze(-1).squeeze(-1).squeeze(0)
        return box_feature

    def cartesian2BEV(self, cartesian_coord, h, w):
    
        left, right, up, down = cartesian_coord
        left_bev = int((left + 140.8) / (281.6 / w))
        right_bev = int((right + 140.8) / (281.6 / w))
        up_bev = int((up + 40) / (80 / h))
        down_bev = int((down + 40) / (80 / h))
        
        return torch.Tensor([left_bev, right_bev, up_bev, down_bev]).cuda()

    def score_func(self, box_features):
        # scores = [self.sigmoid(self.score_linear(box_features[i])) for i in range(len(box_features))]
        # return scores
        scores = self.sigmoid(self.score_linear(cat_list(box_features)))
        return scores
    
    def get_box_bev_coordination(self, boxes, h, w):
        # box_return = deepcopy(boxes)
        box_return = [[[] for j in range(len(boxes[i]))] for i in range(len(boxes))]
        for i in range(len(boxes)):
            boxes_batch = boxes[i]
            for j in range(len(boxes_batch)):
                box_agent = boxes_batch[j]
                temp = []
                for k in range(len(box_agent)):
                    box = box_agent[k]
                    left, right, up, down = box[:,0].min(), box[:,0].max(), box[:,1].min(), box[:,1].max()
                    temp.append(self.cartesian2BEV([left, right, up, down], h, w))
                box_return[i][j] = torch.stack(temp, dim=0)
        return box_return

    def forward(self, data: dict) -> dict:
        _, c, h, w = data['feature'].shape
        features_matching = self.feature_conv(data['feature'].view(-1, c, h, w))
        keypoints = self.get_box_bev_coordination(data['box'], h, w)
        boxes_feature = self.extract_boxes_descriptors(features_matching, keypoints, data['agent_num'], data['box_num'])
        # scores = [self.score_func(boxes_feature[batch]) for batch in range(b)]
        # scores = self.score_func(boxes_feature)
        scores = cat_list(data['scores']).cuda()
        return {
            'keypoints': keypoints,
            'scores': scores,
            'descriptors': boxes_feature,
        }


