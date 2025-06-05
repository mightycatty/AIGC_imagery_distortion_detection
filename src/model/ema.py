# -*- coding: utf-8 -*-
"""
@Time ： 2025/3/18 20:37
@Auth ： heshuai.sec@gmail.com
@File ：ema.py
"""
import torch


class ExponentialMovingAverage:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow_params = [param.clone().detach() for param in model.parameters()]
        self.collected_params = []

    def update(self):
        for shadow_param, param in zip(self.shadow_params, self.model.parameters()):
            shadow_param.data = self.decay * shadow_param.data + (1.0 - self.decay) * param.data

    def apply_shadow(self):
        self.collected_params = [param.clone().detach() for param in self.model.parameters()]
        for shadow_param, param in zip(self.shadow_params, self.model.parameters()):
            param.data.copy_(shadow_param.data)

    def restore(self):
        for collected_param, param in zip(self.collected_params, self.model.parameters()):
            param.data.copy_(collected_param.data)
