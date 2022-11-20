DATA_ROOT=/media/bjw/Disk/depth_data/

# # kitti
# python test.py --config configs/v3/kitti_raw.txt \
#     --ckpt_path ckpts/kitti_scv3/version_3/epoch=75-val_loss=0.1438.ckpt
python test.py --config configs/v1/kitti_raw.txt \
    --ckpt_path ckpts/kitti_scv1/version_0/epoch=99-val_loss=0.1411.ckpt


# # ddad
# python test.py --config configs/v3/ddad.txt \
#     --ckpt_path ckpts/ddad_scv3/version_7/epoch=83-val_loss=0.1469.ckpt
# python test.py --config configs/v1/ddad.txt \
#     --ckpt_path ckpts/ddad_scv1/version_1/epoch=86-val_loss=0.1769.ckpt


# make3d
# python test.py --config configs/v3/ddad.txt --dataset_dir data/make3d \
#     --ckpt_path ckpts/ddad_scv3/version_7/epoch=83-val_loss=0.1469.ckpt
# python test.py --config configs/v1/ddad.txt --dataset_dir data/make3d \
#     --ckpt_path ckpts/ddad_scv1/version_1/epoch=86-val_loss=0.1769.ckpt


# # nyu
# # python test.py --config configs/v3/nyu.txt \
# #     --ckpt_path ckpts/nyu_scv3/version_18/epoch=97-val_loss=0.1398.ckpt
# python test.py --config configs/v1/nyu.txt \
#     --ckpt_path ckpts/nyu_scv1/version_2/epoch=28-val_loss=0.1753.ckpt


# bonn
# python test.py --config configs/v3/bonn.txt \
#     --ckpt_path ckpts/bonn_scv3/version_6/epoch=90-val_loss=0.1342.ckpt
# python test.py --config configs/v1/bonn.txt \
#     --ckpt_path ckpts/bonn_scv1/version_0/epoch=34-val_loss=0.1633.ckpt  


# tum
# python test.py --config configs/v3/tum.txt \
#     --ckpt_path ckpts/tum_scv3/version_1/epoch=99-val_loss=0.1655.ckpt
# python test.py --config configs/v1/bonn.txt \
#     --ckpt_path ckpts/bonn_scv1/version_0/epoch=34-val_loss=0.1633.ckpt  


# # scannet
# python test.py --config configs/v3/nyu.txt --dataset_dir data/scannet --dataset_name bonn \
#     --ckpt_path ckpts/nyu_scv3/version_18/epoch=97-val_loss=0.1398.ckpt
# python test.py --config configs/v1/nyu.txt --dataset_dir data/scannet --dataset_name bonn \
#     --ckpt_path ckpts/nyu_scv1/version_1/epoch=98-val_loss=0.1857.ckpt