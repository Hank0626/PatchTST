# 原版

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers import BertTokenizer, BertModel
from einops import rearrange
from embed import DataEmbedding, DataEmbedding_wo_time
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType

class GPT2ModelWithLabels(GPT2Model):
    def forward(self, input_ids=None, labels=None, **kwargs):
        # 调用父类的forward方法
        outputs = super().forward(input_ids, **kwargs)

        return outputs

class Encoder(nn.Module):
    def __init__(self, input_dim, word_embedding, hidden_dim=768, num_heads=12, num_encoder_layers=1):
        super(Encoder, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)
        
        self.word_embedding = word_embedding.T

    def forward(self, x):
        B = x.shape[0]
        if self.word_embedding.ndim == 2:
            self.word_embedding = self.word_embedding.repeat(B, 1, 1)

        x = self.linear(x)

        x = self.transformer_encoder(x.transpose(0, 1)).transpose(0, 1)

        x_time = x

        q = x.transpose(0, 1)
        k = v = self.word_embedding.transpose(0, 1)
        x, _ = self.cross_attention(q, k, v)

        x = x.transpose(0, 1)

        return x_time, x


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
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=configs.r,
            lora_alpha=configs.lora_alpha,
            lora_dropout=configs.lora_dropout,
            target_modules=["c_attn"]
        )
        
        if configs.is_gpt:
            if configs.pretrain:
                self.gpt2 = GPT2ModelWithLabels.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)  # loads a pretrained GPT-2 base model
                self.gpt2_text = GPT2ModelWithLabels.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)  # loads a pretrained GPT-2 base model
            else:
                print("------------------no pretrain------------------")
                self.gpt2 = GPT2Model(GPT2Config())
            self.gpt2.h = self.gpt2.h[:configs.gpt_layers]
            self.gpt2_text.h = self.gpt2_text.h[:configs.gpt_layers]
            self.gpt2 = get_peft_model(self.gpt2, peft_config)
            # print("gpt2 = {}".format(self.gpt2))

        word_embedding = torch.tensor(torch.load(configs.word_embedding_path)).to(device=device)

        # self.in_layer = nn.Linear(configs.patch_size, configs.d_model)
        self.in_layer = Encoder(configs.seq_len, word_embedding, hidden_dim=configs.d_model)
        self.out_layer = nn.Linear(configs.d_model, configs.pred_len)
        
        if configs.freeze and configs.pretrain:
            for i, (name, param) in enumerate(self.gpt2.named_parameters()):
                if 'ln' in name or 'wpe' in name or 'lora' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            
            for i, (name, param) in enumerate(self.gpt2_text.named_parameters()):
                if 'wpe' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        for layer in (self.gpt2_text, self.gpt2, self.in_layer, self.out_layer):
            layer.to(device=device)
            layer.train()
        
        self.cnt = 0


    def forward(self, x, itr):
        B, L, M = x.shape

        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False)+ 1e-5).detach() 
        x /= stdev

        x = rearrange(x, 'b l m -> b m l')

        # x = self.padding_patch_layer(x)
        # x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        # x = rearrange(x, 'b m n p -> (b m) n p')

        outputs_time1, outputs1 = self.in_layer(x)
        if self.is_gpt:
            outputs_time = self.gpt2(inputs_embeds=outputs_time1).last_hidden_state
            outputs_text = self.gpt2_text(inputs_embeds=outputs1).last_hidden_state

        outputs_time += outputs_time1 
        outputs_text += outputs1
        
        outputs_time = self.out_layer(outputs_time)
        outputs_text = self.out_layer(outputs_text)
        
        outputs_time = rearrange(outputs_time, 'b m l -> b l m')
        outputs_text = rearrange(outputs_text, 'b m l -> b l m')
        
        # outputs = rearrange(outputs, '(b m) l -> b l m', b=B)
        
        outputs_text = outputs_text * stdev + means
        outputs_time = outputs_time * stdev + means

        return outputs_text, outputs_time
