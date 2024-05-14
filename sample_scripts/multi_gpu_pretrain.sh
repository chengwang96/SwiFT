TRAINER_ARGS='--accelerator gpu --max_epochs 10 --precision 16 --num_nodes 1 --devices 2 --strategy DDP' # specify the number of gpus as '--devices'
MAIN_ARGS='--loggername tensorboard --classifier_module v6 --dataset_name S1200 --image_path ./data/HCP1200_MNI_to_TRs_minmax'
DATA_ARGS='--batch_size 4 --num_workers 4 --input_type rest --project_name swift_pretrain'
# DEFAULT_ARGS='--project_name {neptune project name}'
OPTIONAL_ARGS='--c_multiplier 2 --last_layer_full_MSA True --clf_head_version v1 --downstream_task int_total' #--use_scheduler --gamma 0.5 --cycle 0.5'
RESUME_ARGS='--pretraining --use_contrastive --contrastive_type 1'

# export NEPTUNE_API_TOKEN="{neptune API token}"

export CUDA_VISIBLE_DEVICES=2,3
export NCCL_P2P_DISABLE=1

python project/main.py $TRAINER_ARGS $MAIN_ARGS $DEFAULT_ARGS $DATA_ARGS $OPTIONAL_ARGS $RESUME_ARGS \
--dataset_split_num 1 --seed 1 --learning_rate 5e-5 --model swin4d_ver7 --depth 2 2 6 2 --embed_dim 36 --sequence_length 20 --first_window_size 4 4 4 4 --window_size 4 4 4 4 --img_size 96 96 96 20