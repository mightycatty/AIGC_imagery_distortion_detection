import os.path
import random
import time
import io
import torch
import pickle
from transformers import AutoProcessor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image, ImageOps
from skimage.transform import resize
import numpy as np
import cv2
from configs import VALID_SCORES
from tqdm import tqdm
import json


def add_jpeg_noise(img):
    # Randomly add JPEG noise with quality between 70 and 100
    quality = random.randint(70, 100)
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=quality)
    noisy_img = Image.open(io.BytesIO(buffer.getvalue()))
    return noisy_img


class RAHFDataset(Dataset):
    def __init__(self, datapath, data_type, pretrained_processor_path, finetune=False, img_len=448):
        self.img_len = img_len
        # self.tag_word = ['human artifact', 'human mask']  # 使用siglip时需要
        self.tag_word = ['human artifact', 'human segmentation']  # 使用AltCLIP时需要
        self.finetune = finetune
        self.processor = AutoProcessor.from_pretrained(pretrained_processor_path)
        self.processor.image_processor.do_resize = False
        self.processor.image_processor.do_center_crop = False  # 保持图片原大小
        self.to_tensor = transforms.ToTensor()
        self.datapath = datapath
        self.data_type = data_type
        # 加载pkl文件
        self.data_info = self.load_info()
        self.images = []
        self.prompts_en = []
        self.prompts_cn = []
        self.heatmaps = []
        self.scores = []
        self.img_name = list(self.data_info.keys())
        for i in range(len(self.img_name)):
            cur_img = self.img_name[i]
            img = Image.open(f"{self.datapath}/{self.data_type}/images/{cur_img}")
            self.images.append(img.resize((self.img_len, self.img_len), Image.LANCZOS))
            # prompt_cn = self.data_info[cur_img]['prompt_cn']
            prompt_en = self.data_info[cur_img]['prompt']
            self.prompts_en.append(prompt_en)
            # self.prompts_cn.append(prompt_cn)

            artifact_map = self.data_info[cur_img]['heat_map'].astype(float)
            artifact_map = artifact_map / 255.0  # 热力图归一化到0-1

            # misalignment_map = self.data_info[cur_img]['human_mask'].astype(float)  # 使用0-1二值的人体mask时需要
            misalignment_map = np.zeros((512, 512))
            self.heatmaps.append([artifact_map, misalignment_map])

            norm_score = (self.data_info[cur_img]['score'] - 1.0) / 4.0
            self.scores.append((norm_score, 0))  # 人体mask没有分数
            if i % 1000 == 0:
                print(f"Processed {i} images.")

        if data_type == 'train' and self.finetune:
            self.finetune_info = self.load_info(specific_name='finetune')
            self.finetune_images = []
            self.finetune_prompts = []
            self.finetune_heatmaps = []
            self.finetune_scores = []
            self.finetune_img_names = list(self.finetune_info.keys())
            for i in range(len(self.finetune_img_names)):
                cur_img = self.finetune_img_names[i]
                img = Image.open(f"{self.datapath}/{self.data_type}/images/{cur_img}")
                self.finetune_images.append(img.resize((self.img_len, self.img_len), Image.LANCZOS))
                self.finetune_prompts.append(self.finetune_info[cur_img]['prompt'])
                artifact_map = self.finetune_info[cur_img]['artifact_map'].astype(float)
                misalignment_map = self.finetune_info[cur_img]['misalignment_map'].astype(float)
                self.finetune_heatmaps.append([artifact_map, misalignment_map])
                self.finetune_scores.append(
                    (self.finetune_info[cur_img]['artifact_score'], self.finetune_info[cur_img]['misalignment_score']))
                if i % 1000 == 0:
                    print(f"Processed {i} finetuning images.")

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, idx):
        if self.data_type == 'train' and self.finetune and random.random() < 0.5:  # choose finetune image to train with probability of 0.1
            finetune_idx = idx % len(self.finetune_img_names)
            finetune_img_name = self.finetune_img_names[finetune_idx]
            input_img = self.finetune_images[finetune_idx]
            input_prompt = self.finetune_prompts[finetune_idx]
            target_heatmaps = self.finetune_heatmaps[finetune_idx]
            input_img, target_heatmaps, img_pos = self.finetune_augment(input_img, target_heatmaps)
            cur_input = self.processor(images=input_img, text=[f"{self.tag_word[0]} {input_prompt}",
                                                               f"{self.tag_word[1]} {input_prompt}"],
                                       padding="max_length", return_tensors="pt", truncation=True)

            cur_target = {}
            cur_target['artifact_map'] = (self.to_tensor(target_heatmaps[0]))
            cur_target['misalignment_map'] = (self.to_tensor(target_heatmaps[1]))
            cur_target['artifact_score'] = self.finetune_scores[finetune_idx][0]
            cur_target['misalignment_score'] = self.finetune_scores[finetune_idx][1]
            cur_target['img_name'] = finetune_img_name
            cur_target['img_pos'] = torch.tensor(img_pos)
            return cur_input, cur_target

        else:
            img_name = self.img_name[idx]
            input_img = self.images[idx]
            input_prompt = self.prompts_en[idx]
            target_heatmaps = self.heatmaps[idx]
            if self.data_type == 'train':
                input_img, target_heatmaps = self.data_augment(input_img, target_heatmaps)
            cur_input = self.processor(images=input_img, text=[f"{self.tag_word[0]} {input_prompt}",
                                                               f"{self.tag_word[1]} {input_prompt}"],
                                       padding="max_length", return_tensors="pt", truncation=True)
            cur_target = {}
            cur_target['artifact_map'] = (self.to_tensor(target_heatmaps[0]))
            cur_target['misalignment_map'] = (self.to_tensor(target_heatmaps[1]))
            cur_target['artifact_score'] = self.scores[idx][0]
            cur_target['misalignment_score'] = self.scores[idx][1]
            cur_target['img_name'] = img_name
            cur_target['img_pos'] = torch.tensor((0, 0, self.img_len))
            return cur_input, cur_target

    def load_info(self, specific_name=None):
        if specific_name:
            print(f'Loading {specific_name} data info...')
            data_info = pickle.load(open(f'{self.datapath}/{specific_name}_info.pkl', 'rb'))
        else:
            print(f'Loading {self.data_type} data info...')
            data_info = pickle.load(open(f'{self.datapath}/{self.data_type}_info.pkl', 'rb'))
        return data_info

    def data_augment(self, img, heatmaps):
        if random.random() < 0.5:  # 50% chance to crop
            crop_size = int(img.height * random.uniform(0.8, 1.0)), int(img.width * random.uniform(0.8, 1.0))
            crop_region = transforms.RandomCrop.get_params(img, crop_size)
            img = transforms.functional.crop(img, crop_region[0], crop_region[1], crop_region[2], crop_region[3])
            heatmaps = [resize(heatmap, (self.img_len, self.img_len), mode='reflect', anti_aliasing=True)
                        for heatmap in heatmaps]
            heatmaps = [heatmap[crop_region[0]:crop_region[0] + crop_region[2],
                        crop_region[1]:crop_region[1] + crop_region[3]]
                        for heatmap in heatmaps]
            img = img.resize((self.img_len, self.img_len), Image.LANCZOS)
            heatmaps = [resize(heatmap, (512, 512), mode='reflect', anti_aliasing=True)
                        for heatmap in heatmaps]
        data_transforms = transforms.Compose([
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.05, contrast=(0.8, 1), hue=0.025, saturation=(0.8, 1)),
                add_jpeg_noise
            ], p=0.1),
            transforms.RandomApply([transforms.Grayscale(3)], p=0.1)
        ])

        img = data_transforms(img)
        return img, heatmaps

    def finetune_augment(self, img, heatmaps):
        data_transforms = transforms.Compose([
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.05, contrast=(0.8, 1), hue=0.025, saturation=(0.8, 1)),
                add_jpeg_noise
            ], p=0.2),
            transforms.RandomApply([transforms.Grayscale(3)], p=0.2)
        ])
        img = data_transforms(img)
        # rescale gt, do nothing to heatmaps
        if random.random() < 0.9:  # very small image
            scale = random.uniform(0.2, 0.5)
        else:
            scale = random.uniform(0.5, 1.0)
        new_len = int(scale * self.img_len)
        small_img = img.resize((new_len, new_len), Image.LANCZOS)
        top_left_x, top_left_y = random.randint(0, self.img_len - new_len), random.randint(0, self.img_len - new_len)
        pad_left, pad_top = top_left_x, top_left_y
        pad_right, pad_bottom = self.img_len - new_len - pad_left, self.img_len - new_len - pad_top
        pad_color = (255, 255, 255)  # white padding
        pad_img = ImageOps.expand(small_img, border=(pad_left, pad_top, pad_right, pad_bottom), fill=pad_color)
        return pad_img, heatmaps, (top_left_x, top_left_y, new_len)


class RAHFDatasetCustom(Dataset):
    def __init__(self, datapath, data_type='train', img_root=None, pretrained_processor_path=None, finetune=False,
                 img_len=448, mask_size=512,
                 val_split=1000, processor=None, prompt_template='altclip', load_all=False, class_balance=False,
                 args=None):
        self.args = args
        self.img_len = img_len
        self.mask_size = mask_size
        self.val_split = val_split
        self.datapath = datapath
        self.img_root = img_root

        # self.tag_word = ['human artifact', 'human mask']  # 使用siglip时需要
        self.prompt_template = prompt_template
        self.finetune = finetune
        if processor is None:
            assert pretrained_processor_path is not None
            self.processor = AutoProcessor.from_pretrained(pretrained_processor_path)
        else:
            self.processor = processor
        self.processor.image_processor.do_resize = False
        self.processor.image_processor.do_center_crop = False  # 保持图片原大小
        self.to_tensor = transforms.ToTensor()
        self.data_type = data_type
        # 加载pkl文件
        self.load_info()
        self.img_name = list(self.data_info.keys())
        self.dataset = {}
        self.class_balance = class_balance
        self.load_all = load_all
        if load_all:
            self._prepare_dataset()

    def load_info(self):
        train_pickle = os.path.join(self.datapath, 'train_info.pkl')
        train_img_root = os.path.join(self.datapath, 'train/images')

        flickr30k_img_root = os.path.join(self.datapath, 'flickr30k')
        flickr30k_pickle = os.path.join(self.datapath, 'flick30k_negative_dataset.pkl')

        test_img_root = os.path.join(self.datapath, 'val/images')
        test_pkl_root = os.path.join(self.datapath, 'text_6887.pkl')
        test_info_json = os.path.join(self.datapath, 'val/val_info.json')

        self.data_info = {}

        if self.data_type == 'val' and self.val_split <= 0:
            print('using pseudolabel dataset...')
            data_info_buffer = pickle.load(open(test_pkl_root, 'rb'))
            # map to training format
            data_info = {}
            with open(test_info_json, 'r') as f:
                test_info = json.load(f)
            for img_name, values in data_info_buffer.items():
                prompt = test_info[img_name]['prompt_en']
                heatmap = np.uint8(values['pred_area'] * 255)
                img_name = img_name + '.jpg'
                score = values['score']
                data_info[img_name] = {'prompt': prompt, 'heat_map': heatmap, 'score': score}
            for img_name, values in data_info.items():
                img_name = os.path.join(test_img_root, img_name)
                self.data_info[img_name] = values
        else:
            print(f'loading {train_pickle}')
            data_info = pickle.load(open(train_pickle, 'rb'))
            if self.val_split > 0:
                print('splitting dataset...')
                # shuffle dict
                random.seed(getattr(self.args, 'seed', 0))
                data_info = list(data_info.items())
                random.shuffle(data_info)
                if self.data_type == 'val':
                    data_info = dict(data_info[:self.val_split])
                else:
                    data_info = dict(data_info[self.val_split:])

            for img_name, values in data_info.items():
                img_name = os.path.join(train_img_root, img_name)
                self.data_info[img_name] = values

            negative_num = getattr(self.args, 'negative_num', 0)
            if self.data_type == 'train' and negative_num > 0:
                print(f'loading {flickr30k_pickle}')
                data_info = pickle.load(open(flickr30k_pickle, 'rb'))
                data_info = list(data_info.items())
                random.shuffle(data_info)
                data_info = dict(data_info[:negative_num])
                print('negative num:', len(data_info))
                for img_name, values in data_info.items():
                    img_name = os.path.join(flickr30k_img_root, img_name)
                    self.data_info[img_name] = values
            for img_name, values in self.data_info.items():
                values['score'] = (values['score'] - 1.) / 4.
        print('total data num:', len(self.data_info))
        # print('10 samples:\n')
        # random_index = random.sample(range(len(self.data_info)), 10)
        # for index in random_index:
        #     img_name = list(self.data_info.keys())[index]
        #     prompt = self.data_info[img_name]['prompt']
        #     score = self.data_info[img_name]['score']
        #     print(f'{img_name}, prompt:{prompt}, score:{score:.4f}')
        return data_info

    def _make_prompt(self, input_prompt):
        self.tag_word = ['human artifact', 'human segmentation']  # 使用AltCLIP时需要
        if self.prompt_template == 'altclip':
            text = [f"{self.tag_word[0]} {input_prompt}",
                    f"{self.tag_word[1]} {input_prompt}"]
        elif 'florence2' in self.prompt_template:
            text = [
                       f"Here's an AI image generated with prompt: {input_prompt}. \n"
                       f"Identify unusual or implausible areas in the image and rate the implausibility score, focusing on distorted objects, distorted human part, unnatural lighting, or illogical spatial relationships.\n"
                   ] * 2
        return text

    def _prepare_item(self, index):
        img_path = self.img_name[index]
        img = Image.open(img_path)
        img = img.resize((self.img_len, self.img_len), Image.LANCZOS)
        # prompt_cn = self.data_info[cur_img]['prompt_cn']
        prompt_en = self.data_info[img_path]['prompt']

        artifact_map = self.data_info[img_path]['heat_map']
        if artifact_map.shape[0] != self.mask_size:
            artifact_map = cv2.resize(artifact_map, (self.mask_size, self.mask_size))
        artifact_map = artifact_map.astype(float) / 255.0  # 热力图归一化到0-1

        # misalignment_map = self.data_info[cur_img]['human_mask'].astype(float)  # 使用0-1二值的人体mask时需要
        misalignment_map = np.zeros((self.mask_size, self.mask_size))
        heatmap = [artifact_map, misalignment_map]

        # norm_score = (self.data_info[img_path]['score'] - 1.0) / 4.0
        norm_score = self.data_info[img_path]['score']

        # score_class = VALID_SCORES.index(round(self.data_info[img_path]['score'], 5))
        score_class = 0  # TODO: INVALID NUMBER
        return img, prompt_en, heatmap, [norm_score, 0], score_class

    def _prepare_dataset(self):
        for idx in tqdm(range(len(self.img_name)), desc='Preparing dataset'):
            img_name = self.img_name[idx]
            input_img, input_prompt, target_heatmaps, scores, score_class = self._prepare_item(idx)
            self.dataset[img_name] = (input_img, input_prompt, target_heatmaps, scores, score_class)

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, idx):
        img_name = self.img_name[idx]
        # if img_name in self.dataset:
        #     input_img, input_prompt, target_heatmaps, scores, score_class = self.dataset[img_name]
        # else:
        input_img, input_prompt, target_heatmaps, scores, score_class = self._prepare_item(idx)
        # # todo: temporal workaround for OOM error by caching only half the dataset
        # if idx < len(self.img_name) // 2:
        #     self.dataset[img_name] = (input_img, input_prompt, target_heatmaps, scores, score_class)
        if self.data_type == 'train':
            input_img, target_heatmaps = self.data_augment(input_img, target_heatmaps)
        input_prompt = self._make_prompt(input_prompt)
        cur_input = self.processor(images=input_img, text=input_prompt,
                                   padding="max_length", return_tensors="pt", truncation=True)
        # print('prompt:', input_prompt, 'input_ids:', cur_input['input_ids'], 'input_ids_shape:', cur_input['input_ids'].shape)
        cur_target = {}
        cur_target['artifact_map'] = (self.to_tensor(target_heatmaps[0]))
        cur_target['misalignment_map'] = (self.to_tensor(target_heatmaps[1]))
        cur_target['artifact_score'] = scores[0]
        cur_target['artifact_class'] = score_class
        cur_target['misalignment_score'] = scores[1]
        cur_target['img_name'] = os.path.basename(img_name)
        cur_target['img_pos'] = torch.tensor((0, 0, self.img_len))
        return cur_input, cur_target

    def vis_item(self, cur_input, cur_target):
        artifact_map = cur_target['artifact_map'].squeeze(0).numpy()
        cur_img = f"{self.datapath}/{self._image_folder}/images/{cur_target['img_name']}"
        img = Image.open(cur_img)
        img = img.resize((artifact_map.shape[0], artifact_map.shape[1]), Image.LANCZOS)
        img = np.array(img)
        artifact_score = cur_target['artifact_score']
        # draw map on img
        # img = (img * 255).astype(np.uint8)
        artifact_map = (artifact_map * 255).astype(np.uint8)
        artifact_map = cv2.cvtColor(artifact_map, cv2.COLOR_GRAY2RGB)
        img = Image.fromarray(img)
        artifact_map = Image.fromarray(artifact_map)
        print(img.size, artifact_map.size)
        img = Image.blend(img, artifact_map, 0.5)
        img = np.array(img)
        # cv2 draw text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, f'artifact score: {artifact_score:.2f}', (10, 30), font, 1, (0, 0, 255), 2)
        return img

    def data_augment(self, img, heatmaps):
        if random.random() < 0.5:  # 50% chance to crop
            crop_size = int(img.height * random.uniform(0.8, 1.0)), int(img.width * random.uniform(0.8, 1.0))
            crop_region = transforms.RandomCrop.get_params(img, crop_size)
            img = transforms.functional.crop(img, crop_region[0], crop_region[1], crop_region[2], crop_region[3])
            heatmaps = [resize(heatmap, (self.img_len, self.img_len), mode='reflect', anti_aliasing=True)
                        for heatmap in heatmaps]
            heatmaps = [heatmap[crop_region[0]:crop_region[0] + crop_region[2],
                        crop_region[1]:crop_region[1] + crop_region[3]]
                        for heatmap in heatmaps]
            img = img.resize((self.img_len, self.img_len), Image.LANCZOS)
            heatmaps = [resize(heatmap, (self.mask_size, self.mask_size), mode='reflect', anti_aliasing=True)
                        for heatmap in heatmaps]
        data_transforms = transforms.Compose([
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.05, contrast=(0.8, 1), hue=0.025, saturation=(0.8, 1)),
                add_jpeg_noise
            ], p=0.1),
            transforms.RandomApply([transforms.Grayscale(3)], p=0.1),
            # transforms.RandomApply([transforms.RandomHorizontalFlip()], p=getattr(self.args, 'hflip', 0.)),
        ])

        img = data_transforms(img)
        # flip img and heatmaps together
        if random.random() < getattr(self.args, 'hflip', 0.):
            img = transforms.functional.hflip(img)
            heatmaps = [np.fliplr(heatmap).copy() for heatmap in heatmaps]

        return img, heatmaps

    def finetune_augment(self, img, heatmaps):
        data_transforms = transforms.Compose([
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.05, contrast=(0.8, 1), hue=0.025, saturation=(0.8, 1)),
                add_jpeg_noise
            ], p=0.2),
            transforms.RandomApply([transforms.Grayscale(3)], p=0.2)
        ])
        img = data_transforms(img)
        # rescale gt, do nothing to heatmaps
        if random.random() < 0.9:  # very small image
            scale = random.uniform(0.2, 0.5)
        else:
            scale = random.uniform(0.5, 1.0)
        new_len = int(scale * self.img_len)
        small_img = img.resize((new_len, new_len), Image.LANCZOS)
        top_left_x, top_left_y = random.randint(0, self.img_len - new_len), random.randint(0, self.img_len - new_len)
        pad_left, pad_top = top_left_x, top_left_y
        pad_right, pad_bottom = self.img_len - new_len - pad_left, self.img_len - new_len - pad_top
        pad_color = (255, 255, 255)  # white padding
        pad_img = ImageOps.expand(small_img, border=(pad_left, pad_top, pad_right, pad_bottom), fill=pad_color)
        return pad_img, heatmaps, (top_left_x, top_left_y, new_len)
