import argparse
import glob
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from LeReS import RelDepthModel, load_ckpt, SPVCNN_CLASSIFICATION, refine_focal, refine_shift
from path import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Configs for LeReS')
    parser.add_argument('--backbone', default='resnext101', help='Checkpoint path to load')
    parser.add_argument('--load_ckpt', default='weights/res101.pth', help='Checkpoint path to load')
    parser.add_argument('--data-dir', help='Checkpoint path to load')

    args = parser.parse_args()
    return args


def scale_torch(img):
    """
    Scale the image and output it in torch.tensor.
    :param img: input rgb is in shape [H, W, C], input depth/disp is in shape [H, W]
    :param scale: the scale factor. float
    :return: img. [C, H, W]
    """
    if len(img.shape) == 2:
        img = img[np.newaxis, :, :]
    if img.shape[2] == 3:
        transform = transforms.Compose([transforms.ToTensor(),
		                                transforms.Normalize((0.485, 0.456, 0.406) , (0.229, 0.224, 0.225) )])
        img = transform(img)
    else:
        img = img.astype(np.float32)
        img = torch.from_numpy(img)
    return img


def make_shift_focallength_models():
    shift_model = SPVCNN_CLASSIFICATION(input_channel=3,
                                        num_classes=1,
                                        cr=1.0,
                                        pres=0.01,
                                        vres=0.01
                                        )
    focal_model = SPVCNN_CLASSIFICATION(input_channel=5,
                                        num_classes=1,
                                        cr=1.0,
                                        pres=0.01,
                                        vres=0.01
                                        )
    shift_model.eval()
    focal_model.eval()
    return shift_model, focal_model


def reconstruct3D_from_depth(rgb, pred_depth, shift_model, focal_length):
    cam_u0 = rgb.shape[1] / 2.0
    cam_v0 = rgb.shape[0] / 2.0
    pred_depth_norm = pred_depth - pred_depth.min() + 0.5

    dmax = np.percentile(pred_depth_norm, 98)
    pred_depth_norm = pred_depth_norm / dmax

    # # proposed focal length, FOV is 60', Note that 60~80' are acceptable.
    # proposed_scaled_focal = (rgb.shape[0] // 2 / np.tan((60/2.0)*np.pi/180))

    # # recover focal
    # focal_scale_1 = refine_focal(pred_depth_norm, proposed_scaled_focal, focal_model, u0=cam_u0, v0=cam_v0)
    # predicted_focal_1 = proposed_scaled_focal / focal_scale_1.item()
    predicted_focal_1 = focal_length

    # recover shift
    shift_1 = refine_shift(pred_depth_norm, shift_model, predicted_focal_1, cam_u0, cam_v0)
    shift_1 = shift_1 if shift_1.item() < 0.6 else torch.tensor([0.6])
    depth_scale_1 = pred_depth_norm - shift_1.item()

    # # recover focal
    # focal_scale_2 = refine_focal(depth_scale_1, predicted_focal_1, focal_model, u0=cam_u0, v0=cam_v0)
    # predicted_focal_2 = predicted_focal_1 / focal_scale_2.item()

    return shift_1, depth_scale_1


def inference_one_scene(scene_dir, depth_model, shift_model):
    
    folder = Path(scene_dir)
    out_dir = folder/'pseudo_depth'

    if os.path.exists(out_dir):
        files = glob.glob(out_dir/'*')
        for f in files:
            os.remove(f)
    else:
        out_dir.makedirs_p()

    K = np.genfromtxt(folder/'cam.txt').reshape(3,3)
    focal_length = K[0, 0]
    
    rgb_files = sorted(folder.files('*.jpg'))
    
    if len(rgb_files) == len(out_dir.files('*.png')):
        return
    
    for idx, f in enumerate(rgb_files):

        rgb = cv2.imread(f)
        rgb_c = rgb[:, :, ::-1].copy()
  
        A_resize = cv2.resize(rgb_c, (448, 448))
        img_torch = scale_torch(A_resize)[None, :, :, :]

        pred_depth = depth_model.inference(img_torch).cpu().numpy().squeeze()
        pred_depth_ori = cv2.resize(pred_depth, (rgb.shape[1], rgb.shape[0]))

        # recover focal length, shift, and scale-invariant depth
        shift, depth_scaleinv = reconstruct3D_from_depth(rgb, pred_depth_ori, shift_model, focal_length)
        # disp = 1 / depth_scaleinv
        # disp = (disp / disp.max() * 60000).astype(np.uint16)

        depth = depth_scaleinv

        name = out_dir/'{:06d}.png'.format(idx)
        cv2.imwrite(name, (depth/depth.max() * 60000).astype(np.uint16))


if __name__ == '__main__':

    args = parse_args()

    # create depth model
    depth_model = RelDepthModel(backbone=args.backbone)
    depth_model.eval()

    # create shift and focal length model
    shift_model, focal_model = make_shift_focallength_models()

    # load checkpoint
    load_ckpt(args, depth_model, shift_model, focal_model)
    depth_model.cuda()
    shift_model.cuda()
    focal_model.cuda()

    for name in sorted(os.listdir(args.data_dir)):
        if os.path.isdir(args.data_dir+name):
            print(name)
            inference_one_scene(args.data_dir+name, depth_model, shift_model)
