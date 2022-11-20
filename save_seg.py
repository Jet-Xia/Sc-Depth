import os
import glob
from config import get_opts
from path import Path
import numpy as np
import cv2 as cv
from tqdm import tqdm


def inference_one_scene(scene_dir):
    folder = Path(scene_dir)
    out_dir = folder/'seg_mask'

    # seg_mask 文件夹初始化： 清空 seg_mask 文件夹
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

        rgb = cv.imread(f)

        seg_mask = inference(rgb)

        basename = os.path.splitext(os.path.basename(f))[0]
        name = out_dir/'{}.png'.format(basename)
        cv.imwrite(name, seg_mask)


def inference(img):
    retval = cv.hfs.HfsSegment.create(384, 640)
    segimg = retval.performSegmentGpu(img)
    return segimg


if __name__ == '__main__':

    hparams = get_opts()

    for name in sorted(os.listdir(hparams.dataset_dir)):
        if os.path.isdir(hparams.dataset_dir + '/' + name):
            print('Floder name: ', name)
            inference_one_scene(hparams.dataset_dir + '/' + name)