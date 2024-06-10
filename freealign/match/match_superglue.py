import numpy as np
import torch
from opencood.utils.box_utils import project_box3d
from opencood.utils.pose_utils import generate_noise
from opencood.utils.transformation_utils import tfm_to_pose, pose_to_tfm
from freealign.optimize import optimize, optimize_rt, optimize_delta_theta, optimize_rt_raw
from freealign.models.matching import Matching

import os
import copy
import cv2
from tqdm import tqdm
import argparse

def main(data_dict, matching):

    pred = matching(data_dict)
    matches = []
    for agent_match in pred:
        matches_agent = []
        pred_pair = agent_match['matches0'][0]
        for i, j in enumerate(pred_pair):
            if j != -1:
                matches_agent.append((i,j))
        matches.append(matches_agent)
    return matches
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Image pair matching and pose evaluation with SuperGlue',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--data_path', type=str, default='opencood/logs/dairv2x_point_pillar_lidar_max_2023_04_13_13_34_49/test',
        help='Path to data.')
    # opencood/logs/dairv2x_point_pillar_lidar_max_2023_04_13_13_34_49/test
    parser.add_argument(
        '--match_model_ckpt', type=str, default='freealign/superglueckpts/2023_05_08_19_19_56/9.pth',
        help='Path to data.')
        
    parser.add_argument(
        '--nms_radius', type=int, default=4,
        help='SuperPoint Non Maximum Suppression (NMS) radius'
        ' (Must be positive)')

    parser.add_argument(
        '--keypoint_threshold', type=float, default=0.005,
        help='SuperPoint keypoint detector confidence threshold')

    parser.add_argument(
        '--max_keypoints', type=int, default=1024,
        help='Maximum number of keypoints detected by Superpoint'
             ' (\'-1\' keeps all keypoints)')

    parser.add_argument(
        '--layer', type=int, default=0, help='Batch size.')

    parser.add_argument(
        '--superglue', choices={'indoor', 'outdoor'}, default='indoor',
        help='SuperGlue weights')

    parser.add_argument(
        '--sinkhorn_iterations', type=int, default=20,
        help='Number of Sinkhorn iterations performed by SuperGlue')

    parser.add_argument(
        '--match_threshold', type=float, default=0.2,
        help='SuperGlue match threshold')
       
    opt = parser.parse_args()

    error_file_list = []
    error_r_list = []
    error_t_list = []
    defeat_count = 0
    skip_count = 0
    total_count = 0
    error_case = 0
    total_count_align = 0
    file_list = []
    for root, dirs, files in os.walk(os.path.join(opt.data_path, 'featurelist')):
        for file in files:
            file_list.append(os.path.join(root, file))
    
    config = {
        'superbox': {
            'nms_radius': opt.nms_radius,
            'keypoint_threshold': opt.keypoint_threshold,
            'max_keypoints': opt.max_keypoints,
            'init_channels': 64
        },
        'superglue': {
            'weights': opt.superglue,
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
        }
    }

    matching = Matching(config).eval().cuda()
    
    ckpt = torch.load("freealign/superglueckpts/2023_05_08_19_19_56/9.pth")
    matching.load_state_dict(ckpt)
    
    for file in tqdm(file_list):
        
        data_dict = {}
        data_dict['feature'] = torch.load(file)[0].cuda()
        single_pred = torch.load(file.replace('featurelist', 'single_pred'))
        data_dict['match_pair'] = [single_pred['ego_match_pair']]
        data_dict['agent_num'] = [len(single_pred['pred'])]
        data_dict['box'] = [single_pred['pred']]
        data_dict['transform'] = [single_pred['transform']]
        data_dict['box_num'] = [[len(data_dict['box'][0][agent]) for agent in range(data_dict['agent_num'][0])]]
        data_dict['scores'] = [single_pred['scores']]
        
        if len(single_pred['pred']) > 1:
            total_count += len(single_pred['pred'])
            match_result = main(data_dict, matching)
            assert len(single_pred['pred']) == len(match_result) + 1
            for i, agent_match_result in enumerate(match_result):
                if len(agent_match_result) >= 3:
                    R_estimated, T_estimated = optimize_rt_raw(agent_match_result, box_ego = single_pred['pred'][0].mean(1)[:,:2], box_edge = single_pred['pred'][i+1].mean(1)[:,:2])
        
                    theta_gt = np.arccos(single_pred['transform'][0,1,0,0,0])

                    if np.abs(R_estimated[0,0]) > 1:
                        theta_predict = np.arccos(np.sign(R_estimated[0,0]))
                    else:
                        theta_predict = np.arccos(R_estimated[0,0])
                    error_r = np.abs(theta_gt - theta_predict)
                    error_t = np.abs(T_estimated - np.array(single_pred['transform'][0,1,0,0:2,3]))
                    print('error of rotation is {}, error of shift is {}'.format(error_r, error_t))
                    if error_t.mean() > 3:
                        error_case += 1
                    error_r_list.append(error_r)
                    error_t_list.append(error_t)
                    total_count_align += 1
                else:
                    defeat_count += 1
                    
    print('error of rotation is', np.array(error_r_list).mean()*180/np.pi)
    print('error of shift is', np.abs(np.stack(error_t_list, axis=0)).mean(0))
    # print(torch.abs(torch.stack(error_t, dim=0)).mean(0))
    print("successful rate {}".format(1 - (defeat_count/total_count)))
    print("success rate {}".format(1 - (error_case/total_count_align)))