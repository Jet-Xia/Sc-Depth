# Generate frame_index for training SC-Depth methods

# absolute path that contains depth datasets
# DATA_ROOT=/media/bjw/Disk/release_depth_data
# DATA_ROOT=/Users/bjw/Research/depth_data
DATA_ROOT=/media/bjw/disk/depth_data


for dataset in 'nyu'
do
python data_prepare/video_filter.py --dataset_dir $DATA_ROOT/$dataset
done;

# python data_prepare/video_filter.py --dataset_dir $DATA_ROOT/tum