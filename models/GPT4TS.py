# 原版

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers import BertTokenizer, BertModel
from einops import rearrange
from embed import DataEmbedding, DataEmbedding_wo_time
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType
from models.GPT2_arch import AccustumGPT2Model





class Encoder_PCA(nn.Module):
    def __init__(self, input_dim, word_embedding, hidden_dim=768, num_heads=12, num_encoder_layers=1):
        super(Encoder_PCA, self).__init__()
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


class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim=768, num_heads=12, num_encoder_layers=2):
        super(Decoder, self).__init__()
        self.linear = nn.Linear(hidden_dim, output_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

    def forward(self, x):
        x = self.transformer_encoder(x.transpose(0, 1)).transpose(0, 1)
        x = self.linear(x)

        return x

class Encoder_Adaptive(nn.Module):
    def __init__(self, input_dim, word_embedding, hidden_dim=768, num_heads=12, num_encoder_layers=1):
        super(Encoder_Adaptive, self).__init__()
        self.register_buffer('word_embedding', word_embedding)
        self.linear = nn.Linear(input_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.dict_proj = nn.Sequential(nn.Linear(768,1000),nn.GELU(),nn.Linear(1000,500))

        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)



    def forward(self, x):
        B = x.shape[0]
        word_weight_mat = self.dict_proj(self.word_embedding).transpose(-1,-2)
        word_weight_mat =  torch.softmax(word_weight_mat/0.001,dim=-1)
        #word_weight_mat =  F.gumbel_softmax(word_weight_mat.sigmoid(), tau=0.01, hard=True, dim=-1)
        #word_weight_mat = (word_weight_mat/0.01).sigmoid()/word_weight_mat.shape[-1]
        word_embedding = word_weight_mat@self.word_embedding
        word_embedding = word_embedding.repeat(B,1,1)

        x = self.linear(x)

        x = self.transformer_encoder(x.transpose(0, 1)).transpose(0, 1)

        x_time = x

        q = x.transpose(0, 1)
        k = v = word_embedding.transpose(0, 1)
        x_text, _ = self.cross_attention(q, k, v)

        x_text = x_text.transpose(0, 1)

        return x_time, x_text

class Scale(nn.Module):
    def __init__(self, c):
        super(Scale, self).__init__()
        self.time_scale = nn.Parameter(torch.ones((c, 1)), requires_grad=True)
        self.text_scale = nn.Parameter(torch.ones((c, 1)), requires_grad=True)
        
    def forward(self, x, s):
        if s == 'time':
            return x * self.time_scale
        else:
            return x * self.text_scale

class GPT4TS(nn.Module):
    
    def __init__(self, configs, device):
        super(GPT4TS, self).__init__()
        self.is_gpt = configs.is_gpt
        self.patch_size = configs.patch_size
        self.pretrain = configs.pretrain
        self.stride = configs.stride
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1
        self.pred_len = configs.pred_len

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
                self.gpt2 = AccustumGPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)  # loads a pretrained GPT-2 base model
                self.gpt2_text = AccustumGPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)  # loads a pretrained GPT-2 base model
            else:
                print("------------------no pretrain------------------")
                self.gpt2 = GPT2Model(GPT2Config())
            self.gpt2.h = self.gpt2.h[:configs.gpt_layers]
            self.gpt2_text.h = self.gpt2_text.h[:configs.gpt_layers]
            self.gpt2 = get_peft_model(self.gpt2, peft_config)
            # print("gpt2 = {}".format(self.gpt2))

        word_embedding = torch.tensor(torch.load(configs.word_embedding_path)).to(device=device)
        #word_embedding = self.gpt2_text.wte.weight.detach()
        # self.in_layer = nn.Linear(configs.patch_size, configs.d_model)
        self.in_layer = Encoder_PCA(configs.seq_len, word_embedding, hidden_dim=configs.d_model)
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

        self.time_proj = nn.ModuleList([nn.Linear(configs.d_model, configs.d_model,bias=False) for _ in range(configs.gpt_layers+1)])
        
        self.text_proj = nn.ModuleList([nn.Linear(configs.d_model, configs.d_model,bias=False) for _ in range(configs.gpt_layers+1)])

        for layer in (self.gpt2_text, self.gpt2, self.in_layer, self.out_layer, self.time_proj, self.text_proj):
            layer.to(device=device)
            layer.train()
        
        self.cnt = 0
        


    def forward(self, x, itr):
        
        B, L, M = x.shape

        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach() 
        x /= stdev

        x = rearrange(x, 'b l m -> b m l')

        outputs_time1, outputs1 = self.in_layer(x)
        if self.is_gpt:
            outputs_time, intermidiate_feat_time = self.gpt2(inputs_embeds=outputs_time1)
            outputs_text, intermidiate_feat_text = self.gpt2_text(inputs_embeds=outputs1)
        # residue connection
        outputs_time += outputs_time1
        outputs_text += outputs1
        
        intermidiate_feat_time = tuple([self.time_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_time))])
        intermidiate_feat_text = tuple([self.text_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_text))])

        outputs_time = self.out_layer(outputs_time)
        outputs_text = self.out_layer(outputs_text)
        
        outputs_time = rearrange(outputs_time, 'b m l -> b l m')
        outputs_text = rearrange(outputs_text, 'b m l -> b l m')

        outputs_text = outputs_text * stdev + means
        outputs_time = outputs_time * stdev + means

        return {
            'outputs_text': outputs_text,
            'outputs_time':outputs_time,
            'intermidiate_time':intermidiate_feat_time,
            'intermidiate_text':intermidiate_feat_text,
        }
