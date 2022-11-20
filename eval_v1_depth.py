import argparse
import cv2
import numpy as np
import os
from tqdm import tqdm
from path import Path
from imageio import imread
from scipy import sparse

################### Options ######################
parser = argparse.ArgumentParser(description="Evaluation scripts")
parser.add_argument("--dataset", required=True, help="kitti or nyu",
                    choices=['nyu', 'bonn', 'tum', 'kitti', 'ddad', 'scannet'], type=str)
parser.add_argument("--pred_depth", required=True,
                    help="predicted depth folders", type=str)
parser.add_argument("--gt_depth", required=True,
                    help="gt depth folders", type=str)

######################################################
args = parser.parse_args()


def load_sparse_depth(filename):
    sparse_depth = sparse.load_npz(filename)
    depth = np.array(sparse_depth.todense())
    return depth


def pixel_depth_errors(gt, pred: np):
    """Computation of error metrics between predicted and ground truth depths
    Args:
        gt (N): ground truth depth
        pred (N): predicted depth
    """

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    log10 = np.mean(np.abs((np.log10(gt) - np.log10(pred))))

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return


class DepthEval():
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
        pred_depths = []

        """ Get results """
        pred_depths = sorted(Path(args.pred_depth).files("*.png"))  # in *.npy

        """ get gt depths """
        if args.dataset in ['nyu', 'scannet', 'bonn', 'tum']:
            gt_depths = sorted(Path(args.gt_depth).files("*.png"))  # in *.png
        elif args.dataset == 'kitti':
            gt_depths = sorted(Path(args.gt_depth).files("*.npy"))  # in *.npy
        elif args.dataset == 'ddad':
            gt_depths = sorted(Path(args.gt_depth).files("*.npz"))  # in *.npz
        else:
            print('the datset is not support')

        assert (len(pred_depths) == len(gt_depths))

        self.evaluate_depth(gt_depths, pred_depths, eval_mono=True)

    def evaluate_depth(self, gt_depths, pred_depths, eval_mono=True):
        """evaluate depth result
        Args:
            gt_depths: list of gt depth files
            pred_depths: list of predicted depth files
            eval_mono (bool): use median scaling if True
        """
        ratios = []
        print("==> Evaluating depth result...")
        for i in tqdm(range(len(pred_depths))):
            # load predicted depth
            pred_depths[i] = imread(gt_depths[i]).astype(np.float32) / 1000

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

            # gt
            gt_depth = gt_depths[i]
            gt_height, gt_width = gt_depth.shape[:2]
            mask = np.logical_and(gt_depth > self.min_depth, gt_depth < self.max_depth)

            # # resize predicted depth to gt resolution
            pred_depth = cv2.resize(pred_depths[i], (gt_width, gt_height))

            # pre-process
            if args.dataset == 'kitti':
                crop = np.array([0.40810811 * gt_height,  0.99189189 * gt_height,
                                0.03594771 * gt_width, 0.96405229 * gt_width]).astype(np.int32)
                crop_mask = np.zeros(mask.shape)
                crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
                mask = np.logical_and(mask, crop_mask)

            elif args.dataset == 'nyu':
                crop = np.array([45, 471, 41, 601]).astype(np.int32)
                crop_mask = np.zeros(mask.shape)
                crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
                mask = np.logical_and(mask, crop_mask)

            val_pred_depth = pred_depth[mask]
            val_pred_depth = cv2.resize(val_pred_depth, (gt_width, gt_height))
            val_gt_depth = gt_depth[mask]
            val_gt_depth = cv2.resize(val_gt_depth, (gt_width, gt_height))

            # median scaling is used for monocular evaluation
            ratio = 1
            if eval_mono:
                ratio = np.median(val_gt_depth) / np.median(val_pred_depth)
                ratios.append(ratio)
                val_pred_depth *= ratio

            val_pred_depth[val_pred_depth < self.min_depth] = self.min_depth
            val_pred_depth[val_pred_depth > self.max_depth] = self.max_depth

            depth_errors_img = np.array(pixel_depth_errors(val_gt_depth, val_pred_depth))


eval = DepthEval()
eval.main()
