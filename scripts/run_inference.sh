# absolute path that contains all datasets
DATA_ROOT=/media/bjw/disk1/depth_data/

# kitti
# INPUT=$DATA_ROOT/kitti/testing/color
# OUTPUT=results/kitti
# CKPT=ckpts/kitti_scv1/version_0/epoch=99-val_loss=0.1411.ckpt
# CONFIG=configs/v1/kitti_raw.txt
# CKPT=ckpts/kitti_scv3/version_3/epoch=75-val_loss=0.1438.ckpt
# CONFIG=configs/v3/kitti_raw.txt

# # nyu
INPUT=$DATA_ROOT/nyu/testing/color
OUTPUT=results/nyu
# CONFIG=configs/v2/nyu.txt
# CKPT=ckpts/nyu_scv2/version_10/epoch=101-val_loss=0.1580.ckpt
CONFIG=configs/v3/nyu.txt
CKPT=ckpts/nyu_scv3/version_16/epoch=93-val_loss=0.1384.ckpt


# ddad
# INPUT=$DATA_ROOT/ddad/testing/color
# OUTPUT=results/ddad
# CKPT=ckpts/ddad_scv1/version_1/epoch=86-val_loss=0.1769.ckpt
# CONFIG=configs/v1/ddad.txt
# CKPT=ckpts/ddad_scv3/version_6/epoch=99-val_loss=0.1438.ckpt
# CONFIG=configs/v3/ddad.txt


# # bonn
# INPUT=$DATA_ROOT/bonn/testing/color
# OUTPUT=results/bonn
# CKPT=ckpts/bonn_scv1/version_1/epoch=7-val_loss=0.2632.ckpt
# CONFIG=configs/v1/bonn.txt
# CKPT=ckpts/bonn_scv2/version_0/epoch=72-val_loss=0.1976.ckpt
# CONFIG=configs/v2/bonn.txt
# CKPT=ckpts/bonn_scv3/version_6/epoch=90-val_loss=0.1342.ckpt
# CONFIG=configs/v3/bonn.txt


# tum
# INPUT=$DATA_ROOT/tum/testing/color
# OUTPUT=results/tum
# CKPT=ckpts/tum_scv1/version_0/epoch=69-val_loss=0.2525.ckpt
# CONFIG=configs/v1/tum.txt
# CKPT=ckpts/tum_scv2/version_1/epoch=33-val_loss=0.2230.ckpt
# CONFIG=configs/v2/tum.txt
# CKPT=ckpts/tum_scv3/version_8/epoch=88-val_loss=0.1632.ckpt
# CONFIG=configs/v3/tum.txt


# inference depth
python inference.py --config $CONFIG \
--input_dir $INPUT --output_dir $OUTPUT \
--ckpt_path $CKPT --save-depth --save-vis

# # inference pseudo depth
# DATASET=tum
# INPUT=data/$DATASET/testing/color
# OUTPUT=results/$DATASET/pseudo_depth
# python infer_pseudo_depth.py --input_dir $INPUT --output_dir $OUTPUT