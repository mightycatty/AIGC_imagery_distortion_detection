# -*- coding: utf-8 -*-
"""
@Time ： 2025/2/26 15:13
@Auth ： heshuai.sec@gmail.com
@File ：loss.py
"""
import torch
from torch import nn


class DiceLoss(nn.Module):
    def __init__(self, smooth=1., ignore_index=None, soft=False):
        super(DiceLoss, self).__init__()
        self.soft = soft
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
        if self.soft:
            pred_sum = torch.sum(pred * pred, dim=(2, 3))
            target_sum = torch.sum(target * target, dim=(2, 3))
        else:
            pred_sum = torch.sum(pred, dim=(2, 3))
            target_sum = torch.sum(target, dim=(2, 3))
        # 计算Dice相似系数
        dsc = (2. * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)

        # 计算Dice Loss
        loss = 1 - dsc

        # 对所有类别的loss取平均
        loss = loss.mean()

        return loss


# plcc loss and rank loss from https://github.com/QMME/T2VQA/blob/main/train.py
def plcc_loss(y_pred, y):
    y = y.flatten()
    y_pred = y_pred.flatten()
    sigma_hat, m_hat = torch.std_mean(y_pred, unbiased=False)
    y_pred = (y_pred - m_hat) / (sigma_hat + 1e-5)
    sigma, m = torch.std_mean(y, unbiased=False)
    y = (y - m) / (sigma + 1e-5)
    loss0 = torch.nn.functional.mse_loss(y_pred, y) / 4
    rho = torch.mean(y_pred * y)
    loss1 = torch.nn.functional.mse_loss(rho * y_pred, y) / 4
    return ((loss0 + loss1) / 2)


def srcc_loss(y_pred, y):
    y = y.flatten()
    y_pred = y_pred.flatten()
    ranking_loss = torch.nn.functional.relu(
        (y_pred - y_pred.t()) * torch.sign((y.t() - y))
    )
    scale = 1 + torch.max(ranking_loss)
    return (
            torch.sum(ranking_loss) / y_pred.shape[0] / (y_pred.shape[0] - 1) / scale
    )


class MaskLoss(torch.nn.Module):
    def __init__(self, type='mse'):
        super(MaskLoss, self).__init__()
        self.dice = DiceLoss(soft=True if 'soft_dice' in type else False)
        self.mse = torch.nn.MSELoss()
        self.bce = torch.nn.BCELoss()
        self.type = type

    def forward(self, pred, target):
        loss = 0.
        if 'mse' in self.type:
            loss += self.mse(pred, target)
        if 'dice' in self.type:
            loss += self.dice(pred, target)
        if 'bce' in self.type:
            loss += self.bce(pred, target)
        return loss


class ScoreLoss(torch.nn.Module):
    def __init__(self, type='mse', temperature=2.0):
        super(ScoreLoss, self).__init__()
        self.mse = torch.nn.MSELoss()
        self.ce = torch.nn.CrossEntropyLoss()
        self.type = type
        self.temperature = temperature

    def forward(self, pred, target):
        loss = 0.
        if 'mse' in self.type or 'ce' in self.type:
            loss += self.mse(pred, target)
        if 'plcc' in self.type:
            loss += plcc_loss(pred, target)
        if 'srcc' in self.type:
            loss += 0.3 * srcc_loss(pred, target)
        return loss
