export CUDA_VISIBLE_DEVICES=3

# seq_len=96
model=GPT4TS

pred_len=96
percent=100

for gpt_layer in 6
do
for seq_len in 192
do

python main.py \
    --root_path ./datasets/ETT-small/ \
    --data_path ETTh2.csv \
    --model_id ETTh2_$model'_'$gpt_layer'_'$seq_len'_'$pred_len'_'$percent \
    --data ett_h \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len $pred_len \
    --batch_size 256 \
    --decay_fac 0.75 \
    --learning_rate 0.0001 \
    --lradj type1 \
    --train_epochs 15 \
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
    --itr 2 \
    --model $model \
    --cos 1 \
    --tmax 20 \
    --pretrain 1 \
    --is_gpt 1 \
    --r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --patience 5

done
done
