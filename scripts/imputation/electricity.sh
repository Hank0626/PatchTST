export CUDA_VISIBLE_DEVICES=1

seq_len=96
model=GPT4TS

for mask_rate in 0.125 0.25 0.375 0.5
do

python run.py \
    --root_path ./datasets/electricity/ \
    --data_path electricity.csv \
    --is_training 1 \
    --task_name imputation \
    --model_id electricity_mask_$mask_rate \
    --data custom \
    --seq_len $seq_len \
    --mask_rate $mask_rate \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --lradj type1 \
    --train_epochs 20 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --dropout 0.3 \
    --enc_in 7 \
    --c_out 7 \
    --gpt_layers 6 \
    --itr 1 \
    --model $model \
    --tmax 20 \
    --cos 1 \
    --r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --patience 3 \
    --task_loss smooth_l1 \
    --logits_loss l1 \
    --distill_loss l1 \
    --feature_w 0 \
    --logits_w 0 

echo '====================================================================================================================='

done