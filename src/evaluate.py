import json
import os
import pickle

import numpy as np
import torch
from PIL import Image
from scipy.stats import spearmanr


# from utils import cprint as print


def get_plcc_srcc(output_scores, gt_scores):
    # for (output_scores, gt_scores) in zip(output_scores_list, gt_scores_list):
    output_scores = np.array(output_scores)
    gt_scores = np.array(gt_scores)
    # Calculate PLCC (Pearson Linear Correlation Coefficient)
    plcc = np.corrcoef(gt_scores, output_scores)[0, 1]

    # Calculate SRCC (Spearman Rank Correlation Coefficient)
    srcc, _ = spearmanr(gt_scores, output_scores)

    print(f'PLCC: {plcc}')
    print(f'SRCC: {srcc}')


def ignore_edge(heatmap):
    heatmap[0:5, :] = 0  # 顶部边缘
    heatmap[-1:-5, :] = 0  # 底部边缘
    heatmap[:, 0:5] = 0  # 左侧边缘
    heatmap[:, -1:-5] = 0  # 右侧边缘
    return heatmap


def compute_num_params(model):
    import collections
    params = collections.defaultdict(int)
    bytes_per_param = 4
    for name, module in model.named_modules():
        model_name = name.split('.')[0]
        if list(module.parameters()):  # 只处理有参数的模块
            total_params = sum(p.numel() for p in module.parameters())
            memory_usage_mb = (total_params * bytes_per_param) / (1024 * 1024)
            #   print(f"模块: {name}, 参数总量: {total_params}, 显存占用: {memory_usage_mb:.2f} MB")
            params[model_name] += memory_usage_mb

    for k, v in params.items():
        print(k, v, "MB")


def save_pickle(obj, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)


def save_heatmap_mask(input_tensor, threshold, img_name, save_path, process_edge=False):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    vis_path = f'{save_path}_vis'
    if not os.path.exists(vis_path):
        os.makedirs(vis_path)
    input_tensor = torch.where(input_tensor > threshold, 1, 0)
    input_numpy = input_tensor.squeeze(0).cpu().numpy().astype(np.uint8)
    if process_edge:
        input_numpy = ignore_edge(input_numpy)
    vis_numpt = input_numpy * 255
    # Convert to PIL Image
    pil_image = Image.fromarray(input_numpy[0])
    # Save the PIL Image
    pil_image.save(f"{save_path}/{img_name}.png")
    pil_vis = Image.fromarray(vis_numpt[0])
    pil_vis.save(f"{vis_path}/{img_name}.png")


def process_segment_output(outputs):
    normed = torch.softmax(outputs, dim=1)
    foreground = normed[:, 1, :, :]
    binary_mask = (foreground > 0.5).float().squeeze(0)
    return binary_mask


def compute_badcase_detect_rate(output, target):
    if not output:
        return 0
    assert len(output) == len(target), "output and target must have the same length"
    det_count = 0
    for out_score, tar_score in zip(output, target):
        out_score = out_score * 4 + 1
        tar_score = tar_score * 4 + 1
        if tar_score < 3 and out_score < 3:
            det_count += 1

    return det_count / len(output)


def result_vis(img_file, mask, score, save_f):
    import cv2
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


def crop_image(image, crop_size):
    """
    从图像的四个角和中心裁剪给定大小的图像，并返回裁剪后的图像列表。

    :param image: PIL Image对象
    :param crop_size: 裁剪的大小，格式为 (width, height)
    :return: 包含五个裁剪图像的列表
    """
    width, height = image.size
    crop_width, crop_height = crop_size
    resize_shape = (int(crop_size[0] * 1.2), int(crop_size[1] * 1.2))
    image = image.resize(resize_shape)

    # 计算四个角和中心的裁剪区域
    crops = [
        image.crop((0, 0, crop_width, crop_height)),  # 左上角
        image.crop((width - crop_width, 0, width, crop_height)),  # 右上角
        image.crop((0, height - crop_height, crop_width, height)),  # 左下角
        image.crop((width - crop_width, height - crop_height, width, height)),  # 右下角
        image.crop(((width - crop_width) // 2, (height - crop_height) // 2,
                    (width + crop_width) // 2, (height + crop_height) // 2))  # 中心
    ]
    return crops


def generate_submit(args):
    from PIL import Image
    from model.model_final import build_preprocessing_and_model
    gpu = "cuda:0"
    data_root = os.environ.get('DATA_ROOT')
    if args.data == 'val':
        print('generating submit for val...')
        save_root = './submit/val'  # save path of the evaluate results
        info_path = os.path.join(data_root, 'val/val_info.json')
        img_root = os.path.join(data_root, 'val/images')
    elif args.data == 'test':
        print('generating submit for test...')
        save_root = './submit/test'  # save path of the evaluate results
        info_path = os.path.join(data_root, 'test/test_info.json')
        img_root = os.path.join(data_root, 'test/images')
    load_checkpoint = args.load_checkpoint
    save_name = os.path.basename(os.path.dirname(load_checkpoint)) + '_' + os.path.basename(load_checkpoint).split('.')[
        0]
    print('result saved root:', save_root)
    vis_root = os.path.join(save_root, f'{save_name}_vis')
    os.makedirs(save_root, exist_ok=True)
    os.makedirs(vis_root, exist_ok=True)
    heatmap_threshold = args.submit_save_threshold

    val_info = json.load(open(info_path, 'r'))
    name2prompt = {k: v['prompt_en'] for k, v in val_info.items()}

    img_files = os.listdir(img_root)

    processor, model = build_preprocessing_and_model(args)

    print(f'Load checkpoint {args.load_checkpoint}')
    checkpoint = torch.load(f'{args.load_checkpoint}', map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=True)
    model.cuda(gpu)
    model.eval()
    tag_word = ['human artifact', 'human segmentation']

    from tqdm import tqdm
    import pickle
    with torch.no_grad():
        preds = {}
        preds_raw = {}
        for img_file in tqdm(img_files):
            img_name = img_file.split('.')[0]
            prompt = name2prompt[img_name]
            img_path = f'{img_root}/{img_file}'
            img = Image.open(img_path)

            # images = crop_image(img, (448, 448))
            image = img.resize((args.input_size, args.input_size), Image.LANCZOS)

            # get mask
            if args.model == 'altclip':
                text = [f"{tag_word[0]} {prompt}",
                        f"{tag_word[1]} {prompt}"]
            elif 'florence2' in args.model:
                text = [
                           f"Here's an AI image generated with prompt: {prompt}. \n"
                           f"Identify unusual or implausible areas in the image and rate the implausibility score, focusing on distorted objects, distorted human part, unnatural lighting, or illogical spatial relationships.\n"
                       ] * 2
            cur_input = processor(images=image, text=text,
                                  padding="max_length", return_tensors="pt", truncation=True)
            inputs_pixel_values, inputs_ids_im = cur_input['pixel_values'].to(gpu), cur_input['input_ids'][0,
                                                                                    :].unsqueeze(0).to(gpu)
            heatmap, score = model(inputs_pixel_values, inputs_ids_im, need_score=True)
            # heatmap, heatmap_softmax = heatmap
            # score, score_softmax = score
            scores = [score.item()]
            score = float(np.mean(scores))

            ori_heatmap = torch.round(heatmap * 255.0)
            raw_mask = ori_heatmap.squeeze(0).cpu().numpy().astype(np.uint8)
            input_tensor = torch.where(ori_heatmap > heatmap_threshold, 1, 0)
            saved_output_im_map = input_tensor.squeeze(0).cpu().numpy().astype(np.uint8)
            saved_output_im_map = np.squeeze(saved_output_im_map)
            result_vis(img_path, np.squeeze(raw_mask), score, vis_root)
            preds[img_name] = {
                "score": score,
                "pred_area": saved_output_im_map
            }
            preds_raw[img_name] = {
                "score": score,
                "pred_area": np.squeeze(raw_mask),
                # "score_p": np.squeeze(score_softmax.cpu().numpy()),
                # 'mask_p': np.squeeze(heatmap_softmax.cpu().numpy())
            }

    with open(f'{save_root}/{save_name}_thre{heatmap_threshold}.pkl', 'wb') as f:
        pickle.dump(preds, f)
    with open(f'{save_root}/{save_name}_raw.pkl', 'wb') as f:
        pickle.dump(preds_raw, f)


if __name__ == '__main__':
    from configs import get_test_args

    args = get_test_args()
    generate_submit(args)
