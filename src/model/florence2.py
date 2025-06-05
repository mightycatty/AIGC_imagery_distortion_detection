# -*- coding: utf-8 -*-
"""
@Time ： 2025/2/26 18:17
@Auth ： heshuai.sec@gmail.com
@File ：florence2.py
"""
from typing import Optional

import numpy as np
import requests
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM


class Florence2EncoderOnly(torch.nn.Module):
    def __init__(self):
        super().__init__()
        model_path = 'microsoft/Florence-2-large'
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        self.encoder = self.model.language_model.get_encoder()
        del self.model.language_model
        self.config = self.model.config

    def get_input_embeddings(self):
        return self.encoder.get_input_embeddings()

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            pixel_values: torch.FloatTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        image_features = None
        if inputs_embeds is None:
            # 1. Extra the input embeddings
            if input_ids is not None:
                inputs_embeds = self.get_input_embeddings()(input_ids)
            # 2. Merge text and images
            if pixel_values is not None:
                # (batch_size, num_image_tokens, hidden_size)

                image_features = self.model._encode_image(pixel_values) # torch.Size([1, 577, 1024]) for 768* 768 input
                # print(image_features.shape, inputs_embeds.shape) # torch.Size([2, 577, 1024]) torch.Size([2, 1024, 1024])
                inputs_embeds, attention_mask = self.model._merge_input_ids_with_image_features(image_features,
                                                                                                inputs_embeds)
        if inputs_embeds is not None:
            attention_mask = attention_mask.to(inputs_embeds.dtype)
        outputs = self.encoder(
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )['last_hidden_state']
        # print(image_features.shape)
        # print(outputs.shape)
        # outputs = outputs[1:577]
        return image_features, outputs