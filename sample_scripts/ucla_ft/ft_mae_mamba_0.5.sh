TRAINER_ARGS='--accelerator gpu --max_epochs 30 --precision 32 --num_nodes 1 --devices 4 --strategy DDP'
MAIN_ARGS='--loggername tensorboard --classifier_module v6 --dataset_name UCLA --image_path ./data/UCLA_MNI_to_TRs_minmax'
DATA_ARGS='--batch_size 16 --num_workers 16 --input_type rest --project_name abcd2ucla_ft_dx_0.5train_mae_mamba_ratio0.5 --limit_training_samples 0.5'
OPTIONAL_ARGS='--c_multiplier 2 --last_layer_full_MSA True --downstream_task diagnosis --downstream_task_type classification'
RESUME_ARGS='--load_model_path ./output/abcd_mae_mamba_ratio0.5/checkpt-epoch=37-valid_loss=0.37.ckpt --num_classes 4'

export CUDA_VISIBLE_DEVICES=0,4,5,6
export NCCL_P2P_DISABLE=1

python project/main.py $TRAINER_ARGS $MAIN_ARGS $DEFAULT_ARGS $DATA_ARGS $OPTIONAL_ARGS $RESUME_ARGS --dataset_split_num 1 --seed 1 --learning_rate 5e-5 --model swin4d_ver7 --depth 2 2 6 2 --embed_dim 36 --sequence_length 20 --first_window_size 4 4 4 4 --window_size 4 4 4 4 --img_size 96 96 96 20 --use_mamba