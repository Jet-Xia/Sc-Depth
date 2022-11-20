# absolute path that contains all datasets
DATA_ROOT=/media/bjw/disk1/depth_data/

export CUDA_VISIBLE_DEVICES=0

# kitti
# python train.py --config configs/v1/kitti_raw.txt --dataset_dir $DATA_ROOT/kitti

# # ddad
# python train.py --config configs/ddad.txt


# # nyu
python train.py --config configs/v2/nyu.txt --dataset_dir $DATA_ROOT/nyu