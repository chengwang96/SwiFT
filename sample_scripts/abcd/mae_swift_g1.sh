TRAINER_ARGS='--accelerator gpu --max_epochs 10 --precision 32 --num_nodes 1 --devices 1'
MAIN_ARGS='--loggername tensorboard --classifier_module v6 --dataset_name ABCD --image_path ./data/ABCD_MNI_to_TRs_minmax'
DATA_ARGS='--batch_size 8 --num_workers 8 --input_type rest --project_name abcd_mae_swift_ratio0.5'
OPTIONAL_ARGS='--c_multiplier 2 --last_layer_full_MSA True --downstream_task sex' # --use_scheduler --gamma 0.5 --cycle 0.5'
RESUME_ARGS='--pretraining --use_mae --spatial_mask window --time_mask random --mask_ratio 0.5'

export CUDA_VISIBLE_DEVICES=2
export NCCL_P2P_DISABLE=1

python project/main.py $TRAINER_ARGS $MAIN_ARGS $DEFAULT_ARGS $DATA_ARGS $OPTIONAL_ARGS $RESUME_ARGS --dataset_split_num 1 --seed 1 --learning_rate 5e-5 --model swin4d_mae --depth 2 2 6 2 --embed_dim 36 --sequence_length 20 --first_window_size 4 4 4 4 --window_size 4 4 4 4 --img_size 96 96 96 20