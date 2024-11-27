#!/bin/bash
# bash sample_scripts/hcp_ft/ts_abcd2hcp_mamba.sh score_name

# Set default score_name
score_name="MMSE_Score"

# Override with the first argument if provided
if [ ! -z "$1" ]; then
  score_name=$1
fi

# export CUDA_VISIBLE_DEVICES=0,1
export NCCL_P2P_DISABLE=1

# Construct project_name using score_name
project_name="hcp_ts_${score_name}_train1.0_swift"

python project/main.py \
  --accelerator gpu \
  --max_epochs 30 \
  --devices 2 \
  --loggername tensorboard \
  --clf_head_version v3 \
  --dataset_name S1200 \
  --image_path ./data/HCP1200_MNI_to_TRs_minmax \
  --batch_size 16 \
  --num_workers 16 \
  --project_name "$project_name" \
  --limit_training_samples 1.0 \
  --last_layer_full_MSA True \
  --downstream_task int_total \
  --score_name "$score_name" \
  --dataset_split_num 1 \
  --seed 1 \
  --learning_rate 5e-5 \
  --model swin4d_ver7 \
  --depth 2 2 6 2 \
  --embed_dim 36 \
  --sequence_length 20 \
  --first_window_size 4 4 4 4 \
  --window_size 4 4 4 4 \
  --img_size 96 96 96 20