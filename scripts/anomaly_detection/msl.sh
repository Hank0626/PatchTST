export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./datasets/anomaly_detection/MSL \
  --model_id MSL \
  --model GPT4TS \
  --data MSL \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --d_model 768 \
  --d_ff 768 \
  --e_layers 1 \
  --enc_in 55 \
  --c_out 55 \
  --anomaly_ratio 1 \
  --batch_size 128 \
  --train_epochs 1
