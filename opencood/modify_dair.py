import os
root_dir = "opencood/logs/dairv2x_point_pillar_lidar_max_2023_04_13_13_34_49/test/single_pred_pose"
scenario_folders = sorted([os.path.join(root_dir, x)
                            for x in os.listdir(root_dir) if
                            os.path.isdir(os.path.join(root_dir, x))])
for scenario_folder in scenario_folders:
    time_list = sorted([x for x in os.listdir(scenario_folder)])
    for file in time_list:
        os.rename(os.path.join(scenario_folder, file), os.path.join(scenario_folder, file.split('_')[0]+'.pt'))