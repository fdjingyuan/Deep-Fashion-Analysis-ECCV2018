
# coding: utf-8
import pandas as pd
import os
import re
from src.const import base_path

partition = pd.read_csv(base_path + 'Eval/list_eval_partition.txt', skiprows=1, sep='\s+')

category = pd.read_csv(
    base_path + 'Anno/list_category_img.txt', skiprows=1, sep='\s+')
category_type = pd.read_csv(
    base_path + 'Anno/list_category_cloth.txt', skiprows=1, sep='\s+')
category_type['category_label'] = range(1, len(category_type) + 1)
category = pd.merge(category, category_type, on='category_label')

# parse landmarks
with open(base_path + 'Anno/list_landmarks.txt') as f:
    f.readline()
    f.readline()
    values = []
    for line in f:
        info = re.split('\s+', line)
        image_name = info[0].strip()
        clothes_type = int(info[1])
        # 1: ["left collar", "right collar", "left sleeve", "right sleeve", "left hem", "right hem"]
        # 2: ["left waistline", "right waistline", "left hem", "right hem"]
        # 3: ["left collar", "right collar", "left sleeve", "right sleeve", "left waistline", "right waistline", "left hem", "right hem"].
        landmark_postions = [(0, 0)] * 8
        landmark_visibilities = [1] * 8
        landmark_in_pic = [1] * 8
        landmark_info = info[2:]
        if clothes_type == 1:  # upper body
            convert = {0: 0, 1: 1, 2: 2, 3: 3, 4: 6, 5: 7}
        elif clothes_type == 2:
            convert = {0: 4, 1: 5, 2: 6, 3: 7}
        elif clothes_type == 3:
            convert = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}
        for i in convert:
            x = int(landmark_info[i * 3 + 1])
            y = int(landmark_info[i * 3 + 2])
            vis = int(landmark_info[i * 3])
            if vis == 2:
                in_pic = 0
            elif vis == 1:
                in_pic = 1
            else:
                in_pic = 1
            if vis == 2:
                vis = 0
            elif vis == 1:
                vis = 0
            else:
                vis = 1
            landmark_postions[convert[i]] = (x, y)
            landmark_visibilities[convert[i]] = vis
            landmark_in_pic[convert[i]] = in_pic
        tmp = []
        for pair in landmark_postions:
            tmp.append(pair[0])
            tmp.append(pair[1])
        landmark_postions = tmp

        line_value = []
        line_value.extend([image_name, clothes_type])
        line_value.extend(landmark_postions)
        line_value.extend(landmark_visibilities)
        line_value.extend(landmark_in_pic)
        values.append(line_value)

name = ['image_name', 'clothes_type']
name.extend(['lm_lc_x', 'lm_lc_y', 'lm_rc_x', 'lm_rc_y',
             'lm_ls_x', 'lm_ls_y', 'lm_rs_x', 'lm_rs_y',
             'lm_lw_x', 'lm_lw_y', 'lm_rw_x', 'lm_rw_y',
             'lm_lh_x', 'lm_lh_y', 'lm_rh_x', 'lm_rh_y'])

name.extend([
    'lm_lc_vis', 'lm_rc_vis',
    'lm_ls_vis', 'lm_rs_vis',
    'lm_lw_vis', 'lm_rw_vis',
    'lm_lh_vis', 'lm_rh_vis',
])

name.extend([
    'lm_lc_in_pic', 'lm_rc_in_pic',
    'lm_ls_in_pic', 'lm_rs_in_pic',
    'lm_lw_in_pic', 'lm_rw_in_pic',
    'lm_lh_in_pic', 'lm_rh_in_pic',
])

landmarks = pd.DataFrame(values, columns=name)

# attribute
attr = pd.read_csv(base_path + 'Anno/list_attr_img.txt', skiprows=2, sep='\s+', names=['image_name'] + ['attr_%d' % i for i in range(1000)])
attr.replace(-1, 0, inplace=True)

# bbox
bbox = pd.read_csv(base_path + 'Anno/list_bbox.txt', skiprows=1, sep='\s+')

# merge all information
assert (category['category_type'] == landmarks['clothes_type']).all()
landmarks = landmarks.drop('clothes_type', axis=1)
category['category_type'] = category['category_type'] - 1  # 0-based
category['category_label'] = category['category_label'] - 1  # 0-based
info_df = pd.merge(category, landmarks, on='image_name', how='inner')
info_df = pd.merge(info_df, attr, on='image_name', how='inner')
info_df = pd.merge(partition, info_df, on='image_name', how='inner')
info_df = pd.merge(bbox, info_df, on='image_name', how='inner')

info_df.to_csv(base_path + 'info.csv', index=False)
