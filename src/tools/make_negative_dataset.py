# -*- coding: utf-8 -*-
"""
@Time ： 2025/3/10 11:57
@Auth ： heshuai.sec@gmail.com
@File ：make_negative_dataset.py
"""
from dataclasses import dataclass
import numpy as np
import os
import cv2
from tqdm import tqdm
import pickle


@dataclass
class Item:
    heat_map: np.ndarray
    prompt: str
    score: int


def central_crop_image_to_square(image):
    h, w = image.shape[:2]
    if h > w:
        start = (h - w) // 2
        return image[start:start + w, :]
    else:
        start = (w - h) // 2
        return image[:, start:start + h]


def crop_flickr30k():
    img_root = None  # TODO: replace with your own path
    txt_path = None
    os.makedirs(save_path, exist_ok=True)
    for img_name in tqdm(os.listdir(img_root)):
        img_path = os.path.join(img_root, img_name)
        image = cv2.imread(img_path)
        image = central_crop_image_to_square(image)
        save_name = os.path.join(save_path, img_name)
        cv2.imwrite(save_name, image)


def make_from_flick30k():
    img_root = None  # TODO: replace with your own path
    txt_path = None
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()[1:]
    data_info = {}
    for line in tqdm(lines):
        img_name = line.split(',')[0]
        if img_name in data_info:
            continue
        caption = line[len(img_name) + 1:]
        caption = caption.strip(' ')
        img_path = os.path.join(img_root, img_name)
        try:
            image = cv2.imread(img_path)
            heat_map = np.zeros(image.shape[:2])
            data_info[img_name] = Item(heat_map, caption, 5).__dict__
        except Exception:  # catch all exceptions except KeyboardInterrupt
            print('error with image:', img_name)
    with open('flick30k_negative_dataset.pkl', 'wb') as f:
        pickle.dump(data_info, f)


if __name__ == '__main__':
    make_from_flick30k()
