TRAINER_ARGS='--accelerator gpu --max_epochs 30 --precision 32 --num_nodes 1 --devices 1'
MAIN_ARGS='--loggername tensorboard --clf_head_version v3 --dataset_name S1200 --image_path ./data/HCP1200_MNI_to_TRs_minmax'
DATA_ARGS='--batch_size 16 --num_workers 16 --input_type rest --project_name test'
OPTIONAL_ARGS='--c_multiplier 2 --last_layer_full_MSA True --downstream_task int_total'
RESUME_ARGS='--load_model_path ./output/abcd_mae_mamba_ratio0.5/checkpt-epoch=37-valid_loss=0.37.ckpt'

export CUDA_VISIBLE_DEVICES=0
export NCCL_P2P_DISABLE=1

python project/main.py $TRAINER_ARGS $MAIN_ARGS $DEFAULT_ARGS $DATA_ARGS $OPTIONAL_ARGS $RESUME_ARGS --dataset_split_num 1 --seed 1 --learning_rate 5e-5 --model swin4d_mae --depth 2 2 6 2 --embed_dim 48 --sequence_length 20 --first_window_size 4 4 4 4 --window_size 4 4 4 4 --img_size 48 48 48 20 --use_mamba

# Small
# python project/main.py $TRAINER_ARGS $MAIN_ARGS $DEFAULT_ARGS $DATA_ARGS $OPTIONAL_ARGS $RESUME_ARGS --dataset_split_num 1 --seed 1 --learning_rate 5e-5 --model swin4d_mae --depth 2 2 18 2 --embed_dim 96 --sequence_length 20 --first_window_size 4 4 4 4 --window_size 4 4 4 4 --img_size 48 48 48 20 --use_mamba

# Small HR
# python project/main.py $TRAINER_ARGS $MAIN_ARGS $DEFAULT_ARGS $DATA_ARGS $OPTIONAL_ARGS $RESUME_ARGS --dataset_split_num 1 --seed 1 --learning_rate 5e-5 --model swin4d_mae --depth 2 2 18 2 --embed_dim 96 --sequence_length 20 --first_window_size 4 4 4 4 --window_size 4 4 4 4 --img_size 96 96 96 20 --use_mamba

# Base HR
# python project/main.py $TRAINER_ARGS $MAIN_ARGS $DEFAULT_ARGS $DATA_ARGS $OPTIONAL_ARGS $RESUME_ARGS --dataset_split_num 1 --seed 1 --learning_rate 5e-5 --model swin4d_mae --depth 2 2 18 2 --embed_dim 192 --sequence_length 20 --first_window_size 4 4 4 4 --window_size 4 4 4 4 --img_size 96 96 96 20 --use_mamba