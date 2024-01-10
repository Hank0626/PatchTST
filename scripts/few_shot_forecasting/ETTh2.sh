export CUDA_VISIBLE_DEVICES=0

seq_len=96
model=GPT4TS

percent=100

for gpt_layer in 6
do
for pred_len in 192 336 720
do

python run.py \
    --root_path ./datasets/ETT-small/ \
    --data_path ETTh2.csv \
    --is_training 1 \
    --task_name long_term_forecast \
    --model_id ETTh2_$seq_len'_'$pred_len \
    --data ETTh2 \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len $pred_len \
    --batch_size 256 \
    --learning_rate 0.001 \
    --train_epochs 100 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --dropout 0.3 \
    --enc_in 7 \
    --c_out 7 \
    --gpt_layers $gpt_layer \
    --itr 1 \
    --model $model \
    --tmax 20 \
    --cos 1 \
    --r 2 \
    --lora_alpha 8 \
    --lora_dropout 0.1 \
    --patience 10 \
    --percent 10 \
    --task_loss l1

echo '====================================================================================================================='
done
done
