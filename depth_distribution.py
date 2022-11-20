import argparse
import cv2
import numpy as np
from path import Path
from tqdm import tqdm
from imageio.v2 import imread
from scipy import sparse
from matplotlib import pyplot as plt

"""
    depth 分布:
        读取 npz -> 计算分布
    rbg 分布：
        读取 img -> 计算channel分布
    灰度 分布
        读取 img -> rgb-灰度 -> 计算分布
"""

################### Options ######################
parser = argparse.ArgumentParser(description="Distribution scripts")
parser.add_argument("--dataset", required=True, help="dataset name",
                    choices=['nyu', 'bonn', 'tum', 'kitti', 'ddad', 'scannet'], type=str)
parser.add_argument("--depth", default=None,
                    help="depth folders", type=str)
parser.add_argument("--gt", default=None,
                    help="gt folders", type=str)

######################################################
args = parser.parse_args()


# 读取npz
def load_sparse_depth(filename):
    sparse_depth = sparse.load_npz(filename)
    depth = np.array(sparse_depth.todense())
    return depth


class DepthDist():
    def __init__(self):

        self.min_depth = 0.1

        if args.dataset == 'nyu':
            self.max_depth = 10.
        elif args.dataset == 'scannet':
            self.max_depth = 10.
        elif args.dataset == 'bonn':
            self.max_depth = 10.
        elif args.dataset == 'tum':
            self.max_depth = 10.
        elif args.dataset == 'kitti':
            self.max_depth = 80.
        elif args.dataset == 'ddad':
            self.max_depth = 200.

    def main(self):

        if args.depth is not None:
            """ get depths """
            if args.dataset in ['nyu', 'scannet', 'bonn', 'tum']:
                depths = sorted(Path(args.depth).files("*.png"))  # in *.png
            elif args.dataset == 'kitti':
                depths = sorted(Path(args.depth).files("*.npy"))  # in *.npy
            elif args.dataset == 'ddad':
                depths = sorted(Path(args.depth).files("*.npy"))  # in *.npz
            else:
                print('the datset is not support')

        if args.gt is not None:
            """ get depths """
            if args.dataset in ['nyu', 'scannet', 'bonn', 'tum']:
                gts = sorted(Path(args.gt).files("*.png"))  # in *.png
            elif args.dataset == 'kitti':
                gts = sorted(Path(args.gt).files("*.npy"))  # in *.npy
            elif args.dataset == 'ddad':
                gts = sorted(Path(args.gt).files("*.npz"))  # in *.npz
            else:
                print('the datset is not support')

            self.evaluate_depth(depths, gts)

    def evaluate_depth(self, depths, gts):
        """evaluate depth result
        Args:
            depths: list of gt depth files
            eval_mono (bool): use median scaling if True
        """

        val_depths = []
        val_gts = []

        print("==> Evaluating depth result...")
        for i in tqdm(range(0, len(depths)-3850)):
            # load depth
            if args.dataset in ['nyu', 'tum']:
                depths[i] = imread(depths[i]).astype(np.float32) / 5000
            elif args.dataset in ['scannet', 'bonn']:
                depths[i] = imread(depths[i]).astype(np.float32) / 1000
            elif args.dataset == 'kitti':
                depths[i] = np.load(depths[i])
            elif args.dataset == 'ddad':
                depths[i] = np.load(depths[i])
            else:
                print('the datset is not support')

            if args.dataset in ['nyu', 'tum']:
                gts[i] = imread(gts[i]).astype(np.float32) / 5000
            elif args.dataset in ['scannet', 'bonn']:
                gts[i] = imread(gts[i]).astype(np.float32) / 1000
            elif args.dataset == 'kitti':
                gts[i] = np.load(gts[i])
            elif args.dataset == 'ddad':
                gts[i] = load_sparse_depth(gts[i])
            else:
                print('the datset is not support')

            gt = gts[i]
            mask_gt = np.logical_and(gt > self.min_depth,
                                  gt < self.max_depth)
            val_gt = gt[mask_gt]
            val_gt[val_gt < self.min_depth] = self.min_depth
            val_gt[val_gt > self.max_depth] = self.max_depth

            # median scaling
            median_gt = np.median(val_gt)

            # gt
            depth = depths[i]
            height, width = depth.shape[:2]
            mask = np.logical_and(depth > self.min_depth,
                                  depth < self.max_depth)

            # pre-process
            if args.dataset == 'kitti':
                crop = np.array([0.40810811 * height,  0.99189189 * height,
                                0.03594771 * width, 0.96405229 * width]).astype(np.int32)
                crop_mask = np.zeros(mask.shape)
                crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
                mask = np.logical_and(mask, crop_mask)

            elif args.dataset == 'nyu':
                crop = np.array([45, 471, 41, 601]).astype(np.int32)
                crop_mask = np.zeros(mask.shape)
                crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
                mask = np.logical_and(mask, crop_mask)

            val_depth = depth[mask]   # list of valid depth
            val_depth[val_depth < self.min_depth] = self.min_depth
            val_depth[val_depth > self.max_depth] = self.max_depth

            # median scaling
            median = np.median(val_depth)
            ratio = median_gt / median
            val_depth = [ratio * i for i in val_depth]
            val_depths = val_depths + val_depth
            val_gts = val_gts + list(val_gt)

        self.calculate_distribution(type='depth', list=val_gts)

    def calculate_distribution(self, type, list):
        """
        calculate the distribution of matrix
        Args:
            type: depth, img
            list: list of object

        Returns:

        """
        if type == 'depth':
            plt.hist(list, bins=int(self.max_depth), range=[0, self.max_depth])
            plt.show()
        elif type == 'img':
            plt.hist(list, bins=self.max_depth, range=[0, self.max_depth])
            plt.show()
        else:
            print('the type is not support')


eval = DepthDist()
eval.main()
