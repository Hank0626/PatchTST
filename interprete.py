import torch
import numpy as np
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers import GPT2Tokenizer
model = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)
wte = model.wte.state_dict()['weight'].cpu().numpy()
word_embedding_path='./wte_pca_500.pt'
word_embedding = np.array(torch.load(word_embedding_path)).transpose(-1,-2)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
selected_idx_list = [6,10,28,12,7]
batch_idx=16 # 选择第一个cross-attn的样本
cross_attn = torch.load('./cross_attn.pt')[batch_idx] # Lx768
top_k = 5

# for selected_idx in selected_idx_list:
#     pca_vec = word_embedding[selected_idx]
#     # dot_product = np.dot(wte, pca_vec)
#     # matrix_norm = np.linalg.norm(wte, axis=1)
#     # vector_norm = np.linalg.norm(pca_vec)
#     # cos_similarity = dot_product / (matrix_norm * vector_norm)
#     #top_k_indices = np.argsort(cos_similarity)[::-1][:top_k]
#
#     l2_distance = np.linalg.norm(wte - pca_vec, axis=1)
#     top_k_indices = np.argsort(l2_distance)[:top_k]
#
#     token = [tokenizer.decode(i) for i in top_k_indices]
#     print('selected_idx:',selected_idx)
#     print("topK LLM token idx:", top_k_indices)
#     print(token)
#     print('------------------------')


for varients in range(cross_attn.shape[0]):
    #遍历每一个元
    pca_vec = cross_attn[varients] #768
    dot_product = np.dot(wte, pca_vec)
    matrix_norm = np.linalg.norm(wte, axis=1)
    vector_norm = np.linalg.norm(pca_vec)
    cos_similarity = dot_product / (matrix_norm * vector_norm)
    top_k_indices = np.argsort(cos_similarity)[::-1][:top_k]
    token = [tokenizer.decode(i) for i in top_k_indices]
    print('VARENTS:',varients)
    print("topK LLM token idx:", top_k_indices)
    print(token)
    print('------------------------')












