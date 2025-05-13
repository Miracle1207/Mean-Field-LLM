set -x

read -r -d '' training_commands <<EOF
openrlhf.cli.train_sft \
   --max_len 2048 \
   --dataset /home/mqr/code/openrlhf/dataset/Weibo/train  \
   --input_key question \
   --output_key response \
   --train_batch_size 128 \
   --micro_train_batch_size 2 \
   --max_samples 500000 \
   --pretrain /home/mqr/code/llama_factory/language_model/Qwen2-7B-Instruct \
   --save_path ./checkpoint/qwen2-7B-sft-lora-0113\
   --save_steps 10 \
   --logging_steps 1 \
   --eval_steps -1 \
   --zero_stage 2 \
   --max_epochs 1 \
   --bf16 \
   --flash_attn \
   --learning_rate 5e-6 \
   --gradient_checkpointing \
   --load_checkpoint \
   --lora_rank 64 \
   --lora_alpha 64 \
   --use_wandb 1f42766e7327bf8b758147de9b31223c16db1e85
EOF

if [[ ${1} != "slurm" ]]; then
    deepspeed --num_gpus=2 --module $training_commands
fi