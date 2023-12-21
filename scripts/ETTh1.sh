
# export CUDA_VISIBLE_DEVICES=6

# seq_len=336
# model=GPT4TS

# for percent in 100
# do
# for pred_len in 96
# do

# python main.py \
#     --root_path ./datasets/ETT-small/ \
#     --data_path ETTh1.csv \
#     --model_id ETTh1_$model'_'$gpt_layer'_'$seq_len'_'$pred_len'_'$percent \
#     --data ett_h \
#     --seq_len $seq_len \
#     --label_len 168 \
#     --pred_len $pred_len \
#     --batch_size 256 \
#     --lradj type4 \
#     --learning_rate $lr \
#     --train_epochs 10 \
#     --decay_fac 0.5 \
#     --d_model 768 \
#     --n_heads 4 \
#     --d_ff 768 \
#     --dropout 0.3 \
#     --enc_in 7 \
#     --c_out 7 \
#     --freq 0 \
#     --patch_size 16 \
#     --stride 8 \
#     --percent $percent \
#     --gpt_layer 6 \
#     --itr 3 \
#     --model $model \
#     --tmax 20 \
#     --cos 1 \
#     --is_gpt 1

# done
# done

export CUDA_VISIBLE_DEVICES=0

seq_len=96
model=GPT4TS

# pred_len=720
percent=100

for gpt_layer in 6
do
for pred_len in 96 192 336 720
do

python main.py \
    --root_path ./datasets/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_$model'_'$gpt_layer'_'$seq_len'_'$pred_len'_'$percent \
    --data ett_h \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len $pred_len \
    --batch_size 256 \
    --decay_fac 0.5 \
    --learning_rate 0.0005 \
    --lradj type1 \
    --train_epochs 100 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --dropout 0.3 \
    --enc_in 7 \
    --c_out 7 \
    --freq 0 \
    --patch_size 16 \
    --stride 8 \
    --percent $percent \
    --gpt_layer $gpt_layer \
    --itr 1 \
    --model $model \
    --cos 1 \
    --tmax 20 \
    --pretrain 1 \
    --is_gpt 1 \
    --r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --patience 10

done
done
