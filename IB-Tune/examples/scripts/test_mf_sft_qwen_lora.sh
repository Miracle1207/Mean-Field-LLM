set -x

read -r -d '' training_commands <<EOF
openrlhf.cli.train_mf \
   --max_len 2048 \
   --dataset /home/mqr/code/mean_field_lm/data/rumdect/Weibo/train  \
   --input_key question \
   --output_key response \
   --train_batch_size 128 \
   --micro_train_batch_size 2 \
   --max_samples 500000 \
   --pretrain /home/mqr/code/llama_factory/language_model/Qwen2-7B-Instruct \
   --load_checkpoint
   --ckpt_path /home/mqr/code/openrlhf/examples/scripts/ckpt/checkpoints_sft
   --save_path ./checkpoint/qwen2-7B-mf-sft-lora\
   --save_steps 10 \
   --logging_steps 1 \
   --eval_steps -1 \
   --zero_stage 2 \
   --max_epochs 1 \
   --bf16 \
   --flash_attn \
   --learning_rate 1e-6 \
   --load_checkpoint \
   --gradient_checkpointing \
   --lora_rank 64 \
   --lora_alpha 64 \
   --mf_coef_max 0.2 \
   --use_wandb 1f42766e7327bf8b758147de9b31223c16db1e85
EOF

if [[ ${1} != "slurm" ]]; then
    deepspeed --num_gpus=2 --module $training_commands
fi

