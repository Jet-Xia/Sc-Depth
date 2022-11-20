import os
import glob
import torchvision.transforms as transforms
from config import get_opts
from SC_Depth import SC_Depth
from path import Path
import torch
import numpy as np
import cv2
from tqdm import tqdm

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


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
                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        img = transform(img)
    else:
        img = img.astype(np.float32)
        img = torch.from_numpy(img)
    return img


def inference_one_scene(scene_dir, depth_model):

    folder = Path(scene_dir)
    out_dir = folder/'leres_depth'

    # leres_depth 文件夹初始化： 清空 leres_depth 文件夹
    if os.path.exists(out_dir):
        files = glob.glob(out_dir/'*')
        for f in files:
            os.remove(f)
    else:
        out_dir.makedirs_p()

    rgb_files = sorted(folder.files('*.jpg'))

    if len(rgb_files) == len(out_dir.files('*.png')):
        return

    for idx, f in tqdm(enumerate(rgb_files)):

        rgb = cv2.imread(f)
        rgb_c = rgb[:, :, ::-1].copy()

        A_resize = cv2.resize(rgb_c, (448, 448))
        img_torch = scale_torch(A_resize)[None, :, :, :]

        pred_depth = inference(depth_model, img_torch).cpu().numpy().squeeze()
        depth = cv2.resize(pred_depth, (rgb.shape[1], rgb.shape[0]))

        basename = os.path.splitext(os.path.basename(f))[0]
        name = out_dir/'{}.png'.format(basename)
        cv2.imwrite(name, (depth/depth.max() * 60000).astype(np.uint16))


@torch.no_grad()
def inference(depth_model, rgb):
        input = rgb.to(device)
        depth = depth_model(input)
        pred_depth_out = depth - depth.min() + 0.01
        return pred_depth_out


if __name__ == '__main__':

    hparams = get_opts()

    system = SC_Depth(hparams)
    system = system.load_from_checkpoint(hparams.ckpt_path, strict=False)

    depth_model = system.depth_net
    depth_model.to(device)
    depth_model.eval()

    for name in sorted(os.listdir(hparams.dataset_dir)):
        if os.path.isdir(hparams.dataset_dir + '/' + name):
            print(name)
            inference_one_scene(hparams.dataset_dir + '/' + name, depth_model)