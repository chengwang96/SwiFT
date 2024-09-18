TRAINER_ARGS='--accelerator gpu --max_epochs 30 --precision 32 --num_nodes 1 --devices 8 --strategy DDP' # specify the number of gpus as '--devices'
MAIN_ARGS='--loggername tensorboard --classifier_module v6 --dataset_name S1200 --image_path ./data/HCP1200_MNI_to_TRs_minmax'
DATA_ARGS='--batch_size 8 --num_workers 8 --input_type rest --project_name ft_mae_mamba_ratio0.6'
OPTIONAL_ARGS='--c_multiplier 2 --last_layer_full_MSA True --downstream_task int_total' # --use_scheduler --gamma 0.5 --cycle 0.5'
RESUME_ARGS='--load_model_path ./output/mae_mamba_ratio0.6/checkpt-epoch=09-valid_loss=0.22.ckpt'

export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_P2P_DISABLE=1

python project/main.py $TRAINER_ARGS $MAIN_ARGS $DEFAULT_ARGS $DATA_ARGS $OPTIONAL_ARGS $RESUME_ARGS \
--dataset_split_num 1 --seed 1 --learning_rate 5e-5 --model swin4d_ver7 --depth 2 2 6 2 --embed_dim 36 --sequence_length 20 --first_window_size 4 4 4 4 --window_size 4 4 4 4 --img_size 96 96 96 20 --use_mamba