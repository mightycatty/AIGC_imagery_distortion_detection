# -*- coding: utf-8 -*-
"""
@Time ： 2025/3/11 11:26
@Auth ： heshuai.sec@gmail.com
@File ：adjust_outputs.py
"""
import pickle
import numpy as np
import os
from PIL import Image
import cv2


def result_vis(img_file, mask, score, save_f):
    # draw mask on img
    img = Image.open(img_file).convert('RGB')
    img = np.array(img)
    img = cv2.resize(img, (512, 512))
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    mask = cv2.resize(mask, (512, 512))
    img = cv2.addWeighted(img, 0.5, mask, 0.5, 0)
    save_name = os.path.basename(img_file)
    # draw score on image
    cv2.putText(img, f'score: {score:.4f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(save_f, save_name), img)


def adjust_binary_mask(pickle_dir, threshold=40):
    with open(pickle_dir, 'rb') as f:
        data = pickle.load(f)
    for key in data:
        mask = data[key]['pred_area']
        mask = np.where(mask > threshold, 1, 0)
        data[key]['pred_area'] = np.squeeze(mask).astype(np.uint8)
    save_name = pickle_dir.split('.')[0] + f'_thre{threshold}.pkl'
    with open(save_name, 'wb') as f:
        pickle.dump(data, f)


def softmax_p_cleaning(pickle_dir, score_threshold=0.7, mask_threshold=40, vis_save_root='clean_vis'):
    os.makedirs(vis_save_root, exist_ok=True)
    with open(pickle_dir, 'rb') as f:
        data = pickle.load(f)
    data_filter = {}
    for key in data:
        mask = data[key]['pred_area']
        score_p = data[key]['score_p']
        if np.max(score_p) > score_threshold:
            data_filter[key] = {
                'score': (float(np.argmax(score_p)) - 1.) / 12.,
                'pred_area': np.squeeze(mask).astype(np.uint8)
            }
            img_f = r'E:\data\competition\NTIRE2025\test\test\{}.jpg'.format(key)
            result_vis(img_f, data_filter[key]['pred_area'], data_filter[key]['score'], vis_save_root)

    save_name = pickle_dir.split('.')[0] + f'_clean.pkl'
    with open(save_name, 'wb') as f:
        pickle.dump(data_filter, f)


def merge_results(pkl_root_list_or_path):
    if isinstance(pkl_root_list_or_path, str):
        pkl_root = pkl_root_list_or_path
        pkls = [x for x in os.listdir(pkl_root) if x.endswith('.pkl')]
        pkls = [os.path.join(pkl_root, x) for x in pkls]
    elif isinstance(pkl_root_list_or_path, list):
        pkls = pkl_root_list_or_path
    else:
        raise ValueError('pkl_root_list_or_path must be a list or str')
    datas = [pickle.load(open(x, 'rb')) for x in pkls]
    data_keys = datas[0].keys()
    data_info_avg = {}
    for key in data_keys:
        masks = [x[key]['pred_area'] for x in datas]
        masks = [np.squeeze(mask).astype(np.float32) for mask in masks]
        avg_masks = np.mean(np.stack(masks, axis=-1), axis=-1).astype(np.uint8)
        data_info_avg[key] = {
            'score': np.mean([x[key]['score'] for x in datas]),
            'pred_area': avg_masks
        }
    save_pkl = os.path.join(pkl_root, 'merged.pkl')
    with open(save_pkl, 'wb') as f:
        pickle.dump(data_info_avg, f)
    return save_pkl


def threshold2mask_area(pickle_dir):
    with open(pickle_dir, 'rb') as f:
        data = pickle.load(f)
    thresholds_list = list(range(10, 256, 1))
    results = {}
    for key in data:
        mask = data[key]['pred_area']
        # mask = np.where(mask > threshold, 1, 0)
        for threshold in thresholds_list:
            mask_sum = np.sum(mask > threshold)
            if threshold not in results:
                results[threshold] = mask_sum
            else:
                results[threshold] += mask_sum
    xs = []
    ys = []
    for item in results.items():
        xs.append(item[0])
        ys.append(item[1])
    # plot
    import matplotlib.pyplot as plt
    plt.plot(xs, ys)
    plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl_root', type=str, required=True)
    args = parser.parse_args()
    pickle_dir = merge_results(args.pkl_root)
    adjust_binary_mask(pickle_dir, threshold=90)
