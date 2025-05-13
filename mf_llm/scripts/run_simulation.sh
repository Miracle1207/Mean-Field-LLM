#!/bin/bash

# List of file names from the data dictionary

file_names="kangxi sewage weibo liuxiang retirement property apple lobster vice explosion traffic towing"



# List of algorithms
#algs="mf_wo_sft_mf_a mf mf_wo_sft_mf mf_wo_sfta state state_sft"
algs="pre hot"
#algs="mf"
#algs="state_sft"

# Define variables for comment_n and simulation_start
comment_n=200
simulation_start=50
#model_type="7B"
#model_type="gpt-4o-mini"
#model_type="DeepSeek-R1-Distill-Qwen-32B"
model_type="1.5B"
task=""
# Loop over each file_name

for file_name in $file_names; do
    for alg in $algs; do
        echo "Running: python run_mf_batch.py --alg $alg --comment_n $comment_n --simulation_start $simulation_start --file_name $file_name --model $model_type --task $task"
        python run_mf_batch.py --alg "$alg" --comment_n "$comment_n" --simulation_start "$simulation_start" --file_name "$file_name" --model "$model_type" --task "$task"
    done
done