# Generate pseudo-depth for training SC-DepthV3

# absolute path that contains depth datasets
# DATA_ROOT=/media/bjw/Disk/release_depth_data
DATA_ROOT=/data/depth_data

# # kitti
# DATASET=$DATA_ROOT/kitti/training/

# nyu
DATASET=$DATA_ROOT/nyu/training/

# # ddad
# DATASET=$DATA_ROOT/ddad/training/

# generating pseudo depth
python pseudo_depth/save_leres_depth.py \
--backbone resnext101 \
--load_ckpt ckpts/res101.pth \
--dataset_dir $DATASET