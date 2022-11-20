import torch
from config import get_opts, get_training_size
import numpy as np
import cv2 as cv
from imageio.v2 import imread, imwrite

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


def random_sample(B, C, H, W, sample_ratio=0.4):

    mask_A = torch.rand(C, H, W).to(device)
    mask_A[mask_A >= (1 - sample_ratio)] = 1
    mask_A[mask_A < (1 - sample_ratio)] = 0
    idx = torch.randperm(mask_A.nelement())
    mask_B = mask_A.view(-1)[idx].view(mask_A.size())
    mask_A = mask_A.repeat(B, 1, 1).view(B, C, H, W) == 1
    mask_B = mask_B.repeat(B, 1, 1).view(B, C, H, W) == 1

    return mask_A, mask_B


def generate_global_target(depth, mask_A, mask_B, theta=0.15):

    za_gt = depth[mask_A]
    zb_gt = depth[mask_B]
    mask_ignoreb = zb_gt > 1e-8
    mask_ignorea = za_gt > 1e-8
    mask_ignore = mask_ignorea | mask_ignoreb
    za_gt = za_gt[mask_ignore]
    zb_gt = zb_gt[mask_ignore]

    flag1 = za_gt / zb_gt
    flag2 = zb_gt / za_gt
    mask1 = flag1 > 1 + theta
    mask2 = flag2 > 1 + theta
    target = torch.zeros(za_gt.size()).to(device)
    target[mask1] = 1
    target[mask2] = -1

    return target


def cal_ranking_loss(t, tgt):
    loss = 1 - (t - tgt).abs().mean()
    return loss


if __name__ == '__main__':
    pass


