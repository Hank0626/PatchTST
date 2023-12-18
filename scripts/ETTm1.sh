# export CUDA_VISIBLE_DEVICES=6

# seq_len=512
# model=GPT4TS

# # for percent in 100
# # do
# # for pred_len in 96 192 336 720
# # do
# percent=100
# pred_len=96
# python main.py \
#     --root_path ./datasets/ETT-small/ \
#     --data_path ETTm1.csv \
#     --model_id ETTm1_$model'_'$gpt_layer'_'$seq_len'_'$pred_len'_'$percent \
#     --data ett_m \
#     --seq_len $seq_len \
#     --label_len 48 \
#     --pred_len $pred_len \
#     --batch_size 256 \
#     --learning_rate 0.0001 \
#     --train_epochs 10 \
#     --decay_fac 0.75 \
#     --d_model 768 \
#     --n_heads 4 \
#     --d_ff 768 \
#     --dropout 0.3 \
#     --enc_in 7 \
#     --c_out 7 \
#     --freq 0 \
#     --patch_size 16 \
#     --stride 16 \
#     --percent $percent \
#     --gpt_layer 6 \
#     --itr 3 \
#     --model $model \
#     --cos 1 \
#     --is_gpt 1
# # done
# # done

export CUDA_VISIBLE_DEVICES=6

seq_len=512
model=GPT4TS

pred_len=96
percent=100

python main.py \
    --root_path ./datasets/ETT-small/ \
    --data_path ETTm1.csv \
    --model_id ETTm1_$model'_'$gpt_layer'_'$seq_len'_'$pred_len'_'$percent \
    --data ett_m \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len $pred_len \
    --batch_size 256 \
    --decay_fac 0.75 \
    --learning_rate 0.0001 \
    --train_epochs 10 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --dropout 1 \
    --enc_in 7 \
    --c_out 7 \
    --freq 0 \
    --patch_size 16 \
    --stride 8 \
    --percent $percent \
    --gpt_layer 6 \
    --itr 1 \
    --model $model \
    --cos 1 \
    --tmax 20 \
    --pretrain 1 \
    --is_gpt 1 \
    --r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1
