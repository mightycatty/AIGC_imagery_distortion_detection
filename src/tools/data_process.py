import pickle
import os
import json
import cv2
import numpy as np
import sys

data_root = os.environ.get('DATA_ROOT')
train_data_root = os.path.join(data_root, 'train')
val_data_root = os.path.join(data_root, 'val')

'''
将标注数据处理为baseline训练格式 train_info.pkl
注：比赛不提供验证集，若有需要请自行划分, 保存为 val_info.pkl 到同一路径
'''


def overlay_heatmap_opencv(mask, image):
    heatmap = cv2.applyColorMap(mask.astype(np.uint8), cv2.COLORMAP_JET)
    overlayed_image = cv2.addWeighted(image, 0.3, heatmap, 0.7, 0)
    return overlayed_image


train_path = os.path.join(train_data_root, 'images')

vis_path = './data/headmap_vis'  # 热力图可视化保存路径
save_path = './data'  # 训练数据保存路径
info_path = os.path.join(train_data_root, 'train_info.json')  # 'train_info.json'  # 训练集标注数据
with open(info_path, 'r') as f:
    info = json.load(f)
os.makedirs(save_path, exist_ok=True)

vis = True
if vis:
    os.makedirs(vis_path, exist_ok=True)
data_info = {}
from tqdm import tqdm

for k, v in tqdm(info.items()):
    try:
        cur_info = {}
        img_path = os.path.join(train_path, k + '.jpg')
        image = cv2.imread(img_path)
        height, width, _ = image.shape

        prompt = v['prompt_en']
        score = v['mos']
        bbox_info = v['bbox_info']
        part_mask = np.zeros([height, width])
        bbox_infos = []

        for person in bbox_info:
            cur_mask = np.zeros([height, width])
            if not person:
                continue
            else:
                for bbox in person:
                    if bbox['bbox_type'] == 1:
                        top_left, bottom_right = bbox['bbox']
                        cur_mask[top_left['y']:bottom_right['y'], top_left['x']:bottom_right['x']] = 1
                    elif bbox['bbox_type'] == 2:
                        points = bbox['bbox']
                        cv2.fillPoly(cur_mask, [np.array(points)], 1)
            part_mask = part_mask + cur_mask

        error_mask = part_mask.astype(np.uint8)
        error_mask[error_mask == 1] = 64
        error_mask[error_mask == 2] = 127
        error_mask[error_mask == 3] = 180
        error_mask[error_mask == 4] = 255

        # resize for baseline model
        resized_heatmap = cv2.resize(error_mask, (512, 512))
        cur_info['heat_map'] = resized_heatmap
        cur_info['prompt'] = prompt
        cur_info['score'] = score
        data_info[k + '.jpg'] = cur_info
        if vis:
            heatmap = overlay_heatmap_opencv(error_mask, image)
            mask_path = os.path.join(vis_path, k + '.jpg')
            cv2.imwrite(mask_path, heatmap)

    except Exception as e:
        print(k, e)

print(len(data_info))
#
with open(os.path.join(save_path, 'train_info.pkl'), 'wb') as f:
    pickle.dump(data_info, f)
