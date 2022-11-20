import numpy as np
import shutil
from tqdm import tqdm
import torch
from imageio import imread, imwrite
from path import Path
import os


def main():
    input_dir = Path('data/bonn/testing/')
    output_dir = Path('data/bonn/testing/')
    output_dir.makedirs_p()
    (output_dir/'color').makedirs_p()
    (output_dir/'depth').makedirs_p()

    scenes = ['rgbd_bonn_balloon2', 'rgbd_bonn_crowd3', 'rgbd_bonn_person_tracking2', 'rgbd_bonn_synchronous2']
    
    idx = 0
    for scene in scenes:
        scene_dir = input_dir/scene

        image_files = sum([(scene_dir).files('*.{}'.format(ext)) for ext in ['jpg', 'png']], [])
        image_files = sorted(image_files)

        depth_files = sum([(scene_dir/'depth').files('*.{}'.format(ext)) for ext in ['png']], [])
        depth_files = sorted(depth_files)

        for img_file, dep_file in zip(tqdm(image_files), tqdm(depth_files)):
            
            shutil.copyfile(img_file, output_dir/'color/{:06d}.jpg'.format(idx))
            shutil.copyfile(dep_file, output_dir/'depth/{:06d}.png'.format(idx))

            idx += 1
            

if __name__ == '__main__':
    main()