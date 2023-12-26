# 启动这个文件的时候，要去修改utils/tools.py的vali函数的eval()和train()的layer的设置

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import math

from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers import AutoTokenizer
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType
from transformers import BertTokenizer, BertModel
from einops import rearrange
from embed import DataEmbedding, DataEmbedding_wo_time
from transformers.models.gpt2.configuration_gpt2 import GPT2Config

import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose


class GPT2ModelWithLabels(GPT2Model):
    def forward(self, input_ids, labels=None, **kwargs):
        # 调用父类的forward方法
        outputs = super().forward(input_ids, **kwargs)

        return outputs

class GPT4TS(nn.Module):
    
    def __init__(self, configs, device):
        super(GPT4TS, self).__init__()
        self.is_gpt = configs.is_gpt
        self.patch_size = configs.patch_size
        self.pretrain = configs.pretrain
        self.stride = configs.stride
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.patch_num += 1
        
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=8, 
            lora_alpha=32, 
            lora_dropout=0.1,
            target_modules=["c_attn"]
        )

        # (n, l, c) -> transformer encoder -> (n, l, c)
        
        if configs.is_gpt:
            if configs.pretrain:
                self.gpt2 = GPT2ModelWithLabels.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)  # loads a pretrained GPT-2 base model
            else:
                print("------------------no pretrain------------------")
                self.gpt2 = GPT2Model(GPT2Config())
            import pdb; pdb.set_trace()
            self.gpt2.h = self.gpt2.h[:configs.gpt_layers]

            self.gpt2 = get_peft_model(self.gpt2, peft_config)

            print("gpt2 = {}".format(self.gpt2))
        
        self.in_layer1 = nn.Linear(configs.patch_size, configs.d_model//3)
        self.in_layer2 = nn.Linear(configs.patch_size, configs.d_model//3)
        self.in_layer3 = nn.Linear(configs.patch_size, configs.d_model//3)
        # self.in_layer = nn.Linear(configs.seq_len, configs.d_model//3)
        self.out_layer1 = nn.Linear(configs.d_model * self.patch_num // 3, configs.pred_len)
        self.out_layer2 = nn.Linear(configs.d_model * self.patch_num // 3, configs.pred_len)
        self.out_layer3 = nn.Linear(configs.d_model * self.patch_num // 3, configs.pred_len)
        # self.out_layer = nn.Linear(configs.d_model//3, configs.pred_len)
        
        if configs.freeze and configs.pretrain:
            for i, (name, param) in enumerate(self.gpt2.named_parameters()):
                if 'lora' in name or 'ln' in name or 'wpe' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        for layer in (self.gpt2, self.in_layer1, self.out_layer1, self.in_layer2, self.out_layer2, self.in_layer3, self.out_layer3):
            layer.to(device=device)
            layer.train()
        
        self.cnt = 0
        self.device = device

    def norm(self, x):
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False)+ 1e-5).detach() 
        x /= stdev
        
        return x, means, stdev

    def forward(self, x, itr):
        B, L, M = x.shape

        x = rearrange(x, 'b l m -> b m l').squeeze(1).cpu().numpy()
        decomp = seasonal_decompose(x.T, period=24, extrapolate_trend='freq')
        x_seasonal, x_trend, x_residual = decomp.seasonal.T, decomp.trend.T, decomp.resid.T
        
        x_seasonal = torch.tensor(x_seasonal, dtype=torch.float).to(self.device).unsqueeze(2)
        x_trend = torch.tensor(x_trend, dtype=torch.float).to(self.device).unsqueeze(2)
        x_residual = torch.tensor(x_residual, dtype=torch.float).to(self.device).unsqueeze(2)

        x_seasonal, mean_seasonal, std_seasonal = self.norm(x_seasonal)
        x_trend, mean_trend, std_trend = self.norm(x_trend)
        x_residual, mean_residual, std_residual = self.norm(x_residual)
        
        x_seasonal = rearrange(x_seasonal, 'b l m -> b m l')
        x_trend = rearrange(x_trend, 'b l m -> b m l')
        x_residual = rearrange(x_residual, 'b l m -> b m l')

        x_seasonal = self.padding_patch_layer(x_seasonal)
        x_seasonal = x_seasonal.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        
        x_trend = self.padding_patch_layer(x_trend)
        x_trend = x_trend.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        
        x_residual = self.padding_patch_layer(x_residual)
        x_residual = x_residual.unfold(dimension=-1, size=self.patch_size, step=self.stride)

        x_seasonal = rearrange(x_seasonal, 'b m n p -> (b m) n p')
        x_trend = rearrange(x_trend, 'b m n p -> (b m) n p')
        x_residual = rearrange(x_residual, 'b m n p -> (b m) n p')

        outputs_trend = self.in_layer1(x_trend)
        outputs_seasonal = self.in_layer2(x_seasonal)
        outputs_residual = self.in_layer3(x_residual)

        outputs = torch.cat([outputs_trend, outputs_seasonal, outputs_residual], dim=2)

        if self.is_gpt:
            outputs = self.gpt2(inputs_embeds=outputs).last_hidden_state

        outputs = outputs.reshape(B*M, -1)
        
        sep = outputs.shape[-1] // 3
        
        outputs_trend1 = self.out_layer1(outputs[:, :sep])
        outputs_seasonal1 = self.out_layer2(outputs[:, sep:2*sep])
        outputs_residual1 = self.out_layer3(outputs[:, 2*sep:3*sep])

        outputs_trend1 = rearrange(outputs_trend1, '(b m) l -> b l m', b=B) * std_trend + mean_trend
        outputs_seasonal1 = rearrange(outputs_seasonal1, '(b m) l -> b l m', b=B) * std_seasonal + mean_seasonal
        outputs_residual1 = rearrange(outputs_residual1, '(b m) l -> b l m', b=B) * std_residual + mean_residual

        # return outputs
        return outputs_trend1 + outputs_seasonal1 + outputs_residual1
