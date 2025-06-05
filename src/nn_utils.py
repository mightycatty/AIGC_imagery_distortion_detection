# -*- coding: utf-8 -*-
"""
@Time ： 2025/2/24 16:29
@Auth ： heshuai.sec@gmail.com
@File ：nn_utils.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1., ignore_index=None):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        """
        计算Dice Loss。
        参数：
            pred: (batch_size, num_classes, height, width)，模型输出的logits或概率
            target: (batch_size, height, width)，真实标签
        返回：
            loss: 标量，Dice Loss值
        """

        # 处理ignore_index
        if self.ignore_index is not None:
            mask = target != self.ignore_index
            target = target[mask]
            pred = pred[mask]

        # 计算交集
        intersection = torch.sum(pred * target, dim=(2, 3))

        # 计算并集
        pred_sum = torch.sum(pred, dim=(2, 3))

        target_sum = torch.sum(target, dim=(2, 3))

        # 计算Dice相似系数
        dsc = (2. * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)

        # 计算Dice Loss
        loss = 1 - dsc

        # 对所有类别的loss取平均
        loss = loss.mean()

        return loss