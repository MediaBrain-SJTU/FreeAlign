import torch

from .superpoint import SuperPoint
from .superglue import SuperGlue
from .superbox import SuperBox
from .superbevglue import SuperBEVGlue
import time

class Matching(torch.nn.Module):
    """ Image Matching Frontend (SuperPoint + SuperGlue) """
    def __init__(self, config={}):
        super().__init__()
        self.superbox = SuperBox(config.get('superbox', {}))
        # self.superpoint = SuperPoint(config.get('superpoint', {}))
        self.superbevglue = SuperBEVGlue(config.get('superglue', {}))

    def forward(self, data):
        """ Run SuperPoint (optionally) and SuperGlue
        SuperPoint is skipped if ['keypoints0', 'keypoints1'] exist in input
        Args:
          data: dictionary with minimal keys: ['image0', 'image1']
        """
        pred = {}
        
        # Extract SuperPoint (keypoints, scores, descriptors) if not provided
        # if 'keypoints0' not in data:
        #     pred0 = self.superbox(data)
        #     pred = {**pred, **{k+'0': v for k, v in pred0.items()}}
        # if 'keypoints1' not in data:
        #     pred1 = self.superbox({'image': data['image1']})
        #     pred = {**pred, **{k+'1': v for k, v in pred1.items()}}
        time_0 = time.time()
        pred = self.superbox(data)
        time_1 = time.time()
        # print('time for box feature extraction is {}'.format(time_1-time_0))

        # Batch all features
        # We should either have i) one image per batch, or
        # ii) the same number of local features for all images in the batch.
        data = {**data, **pred}

        # for k in data:
        #     if isinstance(data[k], (list, tuple)):
        #         data[k] = torch.stack(data[k])
        # data = self.process_data(data)
        # # Perform the matching
        result = []
        load_list = pair_data(data)
        for i in range(len(load_list)):
            result.append(self.superbevglue(load_list[i]))
        # pred = {**pred, **self.superbevglue(data)}
        time_2 = time.time()
        # print('time for box matching is {}'.format(time_2-time_1))
        return result
    
    def process_data(self, data):
        for k in data:
            if isinstance(data[k], (list, tuple)):
                data[k] = torch.stack(data[k])
        return data
    

# def split_data(data, num_agent, box_num):
#     bs = len(num_agent)
#     split_data = []
#     load_list = []
#     count = 0
#     for batch in range(bs):
#         batch_list = []
#         for agent in range(num_agent[batch]):
#             if agent == 0:
#                 ego_feature = data['descriptors'][batch][agent][:, :, count:count+box_num[batch][agent]]
#                 ego_kpts = data['keypoints'][:, :, count:count+box_num[batch][agent]]
#                 ego_scores = data['scores'][:, count:count+box_num[batch][agent]]
#             else:
#                 batch_list.append([ego_feature.clone(), data['descriptors'][:, :, count:count+box_num[batch][agent]], ego_kpts.clone(), data['keypoints'][:, :, count:count+box_num[batch][agent]], ego_scores.clone(), data['scores'][:, count:count+box_num[batch][agent]], data['feature'].shape[-2], data['feature'].shape[-1]])
#                 load_list.append(batch_list[-1])
#             count += box_num[batch][agent]
#         split_data.append(batch_list)
    
#     return split_data, load_list

def pair_data(data: dict):
    load_list = []
    bs = len(data['agent_num'])
    count = 0
    for batch in range(bs):
        batch_list = []
        for agent in range(data['agent_num'][batch]):

            if agent == 0:
                ego_feature = data['descriptors'][batch][agent].unsqueeze(0)
                ego_kpts = data['keypoints'][batch][agent].unsqueeze(0)
                ego_scores = data['scores'][count:count+data['box_num'][batch][agent]]
            else:
                # if len(data['match_pair'][batch][agent-1]) == 0:
                #     continue
                # else:
                if 1:
                    batch_list.append([ego_feature.clone(), data['descriptors'][batch][agent].unsqueeze(0), ego_kpts.clone(), data['keypoints'][batch][agent].unsqueeze(0), ego_scores.clone(), data['scores'][count:count+data['box_num'][batch][agent]], data['feature'].shape[-2], data['feature'].shape[-1], data['match_pair'][batch][agent-1]])
                    load_list.append(batch_list[-1])
            count += data['box_num'][batch][agent]
    return load_list