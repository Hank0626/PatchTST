export CUDA_VISIBLE_DEVICES=0

python -u run.py \
    --task_name anomaly_detection \
    --is_training 1 \
    --root_path ./datasets/anomaly_detection/PSM \
    --model_id PSM \
    --model GPT4TS \
    --data PSM \
    --features M \
    --seq_len 100 \
    --pred_len 0 \
    --d_model 768 \
    --d_ff 768 \
    --e_layers 1 \
    --enc_in 38 \
    --c_out 38 \
    --anomaly_ratio 1 \
    --batch_size 128 \
    --train_epochs 10 \
    --r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --task_loss l1 \
    --distill_loss l1 \
    --logits_loss l1 \
    --task_w 1 \
    --logits_w 0.1 \
    --feature_w 0.01

echo '====================================================================================================================='
