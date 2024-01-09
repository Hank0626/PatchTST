export CUDA_VISIBLE_DEVICES=0

for data in 'EthanolConcentration' 'FaceDetection' 'Heartbeat' 'Heartbeat' 'JapaneseVowels' 'PEMS-SF' 'SelfRegulationSCP1' 'SelfRegulationSCP2' 'SpokenArabicDigits' 'UWaveGestureLibrary'
do

python -u run.py \
    --task_name classification \
    --is_training 1 \
    --root_path ./datasets/classification/$data \
    --model_id $data \
    --model GPT4TS \
    --data UEA \
    --batch_size 512 \
    --d_model 768 \
    --d_ff 768 \
    --n_heads 4 \
    --itr 1 \
    --learning_rate 0.0001 \
    --r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --train_epochs 30 \
    --patience 10 \
    --gpt_layers 6 \
    --task_loss ce \
    --logits_loss l1 \
    --distill_loss l1

echo '====================================================================================================================='
done