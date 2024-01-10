export CUDA_VISIBLE_DEVICES=0

seq_len=96
model=GPT4TS

for mask_rate in 0.125 0.25 0.375 0.5
do

python run.py \
    --root_path ./datasets/ETT-small/ \
    --data_path ETTh2.csv \
    --is_training 1 \
    --task_name imputation \
    --model_id ETTh2_mask_$mask_rate \
    --data ETTh2 \
    --seq_len $seq_len \
    --mask_rate $mask_rate \
    --batch_size 256 \
    --learning_rate 0.0005 \
    --lradj type1 \
    --train_epochs 100 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --dropout 0.3 \
    --enc_in 7 \
    --c_out 7 \
    --gpt_layers 6 \
    --itr 1 \
    --model $model \
    --r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --patience 10 \
    --task_loss l1 \
    --logits_loss l1 \
    --distill_loss l1 \
    --feature_w 0 \
    --logits_w 0

echo '====================================================================================================================='

done