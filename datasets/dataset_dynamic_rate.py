import argparse
import cv2
import numpy as np
import os
from tqdm import tqdm
from path import Path
from imageio import imread
from scipy import sparse

################### Options ######################
parser = argparse.ArgumentParser(description="Dynamic rate scripts")
parser.add_argument("--dataset", required=True, help="kitti or nyu",
                    choices=['nyu', 'bonn', 'tum', 'kitti', 'ddad', 'scannet'], type=str)
parser.add_argument("--depth_dir", required=True,
                    help="training depth folders", type=str)
parser.add_argument("--seg_mask", default=None,
                    help="segmentation mask folders", type=str)

######################################################
args = parser.parse_args()


def load_sparse_depth(filename):
    sparse_depth = sparse.load_npz(filename)
    depth = np.array(sparse_depth.todense())
    return depth


def main():

    min_depth = 0.1

    if args.dataset == 'nyu':
        max_depth = 10.
    elif args.dataset == 'scannet':
        max_depth = 10.
    elif args.dataset == 'bonn':
        max_depth = 10.
    elif args.dataset == 'tum':
        max_depth = 10.
    elif args.dataset == 'kitti':
        max_depth = 80.
    elif args.dataset == 'ddad':
        max_depth = 200.

    """ get gt depths """
    if args.dataset in ['nyu', 'scannet', 'bonn', 'tum']:
        gt_depths = sorted(Path(args.depth_dir).files("*.png"))  # in *.png
    elif args.dataset == 'kitti':
        gt_depths = sorted(Path(args.depth_dir).files("*.npy"))  # in *.npy
    elif args.dataset == 'ddad':
        gt_depths = sorted(Path(args.depth_dir).files("*.npz"))  # in *.npz
    else:
        print('the datset is not support')
    assert len(gt_depths) != 0

    """ Get segmentation masks """
    dynamic_colors = np.loadtxt(
        Path(args.seg_mask) / 'dynamic_colors.txt').astype('uint8')
    seg_masks = sorted(Path(args.seg_mask).files("*.png"))

    for i in tqdm(range(len(gt_depths))):
        # load gt depth
        if args.dataset in ['nyu', 'tum']:
            gt_depths[i] = imread(gt_depths[i]).astype(np.float32) / 5000
        elif args.dataset in ['scannet', 'bonn']:
            gt_depths[i] = imread(gt_depths[i]).astype(np.float32) / 1000
        elif args.dataset == 'kitti':
            gt_depths[i] = np.load(gt_depths[i])
        elif args.dataset == 'ddad':
            gt_depths[i] = load_sparse_depth(gt_depths[i])
        else:
            print('the datset is not support')

        dynamic_mask = np.zeros_like(gt_depths[i])
        seg_mask = imread(seg_masks[i])
        for item in dynamic_colors:

            cal_mask_0 = seg_mask[:, :, 0] == item[0]
            cal_mask_1 = seg_mask[:, :, 1] == item[1]
            cal_mask_2 = seg_mask[:, :, 2] == item[2]
            cal_mask = cal_mask_0 * cal_mask_1 * cal_mask_2
            dynamic_mask[cal_mask] = 1

if __name__ == '__main__':
    main()