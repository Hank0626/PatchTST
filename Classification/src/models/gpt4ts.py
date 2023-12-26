from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from transformers import GPT2ForSequenceClassification
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers import BertTokenizer, BertModel
from einops import rearrange
from .embed import DataEmbedding, DataEmbedding_wo_time
from transformers import AutoTokenizer
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType
from .GPT2_arch import AccustumGPT2Model


class Encoder_PCA(nn.Module):
    def __init__(self, input_dim, word_embedding, prompt_embedding, hidden_dim=768, num_heads=12, num_encoder_layers=1):
        super(Encoder_PCA, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)

        self.word_embedding = word_embedding.T

        self.prompt_embedding = prompt_embedding

    def forward(self, x):
        B = x.shape[0]
        if self.word_embedding.ndim == 2:
            word_embedding = self.word_embedding.repeat(B, 1, 1)

        if self.prompt_embedding.shape[0] != B:
            prompt_embedding = self.prompt_embedding.repeat(B, 1, 1)

        x = self.linear(x)

        x = torch.cat([prompt_embedding, x], dim=1)

        x = self.transformer_encoder(x.transpose(0, 1)).transpose(0, 1)

        x_time = x

        q = x.transpose(0, 1)
        k = v = word_embedding.transpose(0, 1)
        x, _ = self.cross_attention(q, k, v)

        x = x.transpose(0, 1)

        return x_time, x



class gpt4ts(nn.Module):
    
    def __init__(self, config, data,device):
        super(gpt4ts, self).__init__()
        self.pred_len = 0
        self.seq_len = data.max_seq_len
        self.max_len = data.max_seq_len
        self.patch_size = config['patch_size']
        self.stride = config['stride']
        self.gpt_layers = 6
        self.feat_dim = data.feature_df.shape[1]
        self.num_classes = len(data.class_names)
        self.d_model = config['d_model']

        self.patch_num = (self.seq_len - self.patch_size) // self.stride + 1

        #self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.patch_num += 1
        #self.enc_embedding = DataEmbedding(self.feat_dim * self.patch_size, config['d_model'], config['dropout'])

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=config['r'],
            lora_alpha=config['lora_alpha'],
            lora_dropout=config['lora_dropout'],
            target_modules=["c_attn"]
        )


        self.gpt2 = AccustumGPT2Model.from_pretrained('gpt2', output_attentions=True,
                                                              output_hidden_states=True)  # loads a pretrained GPT-2 base model
        self.gpt2_text = AccustumGPT2Model.from_pretrained('gpt2', output_attentions=True,
                                                                   output_hidden_states=True)  # loads a pretrained GPT-2 base model

        self.gpt2.h = self.gpt2.h[:self.gpt_layers]
        self.gpt2_text.h = self.gpt2_text.h[:self.gpt_layers]
        self.gpt2 = get_peft_model(self.gpt2, peft_config)

        prompt = 'Time series classifaction'#prompt_dict[configs.data_path.split('.')[0]]

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        prompt_embedding = self.gpt2.wte(tokenizer.encode(prompt, return_tensors="pt")).mean(1).to(device=device)

        word_embedding = torch.tensor(torch.load(config['word_embedding_path'])).to(device=device)
        # self.in_layer = nn.Linear(configs.patch_size, configs.d_model)
        self.in_layer = Encoder_PCA(self.seq_len, word_embedding, prompt_embedding, hidden_dim=self.d_model)
        self.act = F.gelu
        self.dropout = nn.Dropout(0.1)
        self.ln_proj = nn.LayerNorm(self.feat_dim*self.d_model)
        self.out_layer = nn.Linear(self.feat_dim*self.d_model, self.num_classes)
        self.time_proj = nn.ModuleList([nn.Linear(self.d_model, self.d_model, bias=False) for _ in range(self.gpt_layers + 1)])
        self.text_proj = nn.ModuleList([nn.Linear(self.d_model, self.d_model, bias=False) for _ in range(self.gpt_layers + 1)])
        

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


        for layer in (self.gpt2_text, self.gpt2, self.in_layer, self.out_layer, self.ln_proj, self.time_proj, self.text_proj):
            layer.to(device=device)
            layer.train()



        
    def forward(self, x_enc, x_mark_enc):
        B, L, M = x_enc.shape
        
        input_x = rearrange(x_enc, 'b l m -> b m l')
        outputs_time1, outputs1 = self.in_layer(input_x)

        outputs_time, intermidiate_feat_time = self.gpt2(inputs_embeds=outputs_time1)
        outputs_text, intermidiate_feat_text = self.gpt2_text(inputs_embeds=outputs1)

        outputs_time += outputs_time1
        outputs_text += outputs1

        intermidiate_feat_time = tuple([self.time_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_time))])
        intermidiate_feat_text = tuple([self.text_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_text))])

        outputs_time = outputs_time[:, -M:, :]
        outputs_text = outputs_text[:, -M:, :]

        outputs_time = self.act(outputs_time).reshape(B, -1)
        outputs_time = self.ln_proj(outputs_time)

        outputs_text = self.act(outputs_text).reshape(B, -1)
        outputs_text = self.ln_proj(outputs_text)

        outputs_time = self.out_layer(outputs_time)
        outputs_text = self.out_layer(outputs_text)

        return {
            'outputs_text': outputs_text,
            'outputs_time': outputs_time,
            'intermidiate_time': intermidiate_feat_time,
            'intermidiate_text': intermidiate_feat_text,
        }

    