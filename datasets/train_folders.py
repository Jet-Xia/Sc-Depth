import torch.utils.data as data
import numpy as np
from imageio.v2 import imread
from path import Path
import random
import os


def load_as_float(path):
    return imread(path).astype(np.float32)


# Training Dataset
class TrainFolder(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
        root/scene_1/0000000.jpg
        root/scene_1/0000001.jpg
        ..
        root/scene_1/cam.txt
        root/scene_2/0000000.jpg
        .
        transform functions must take in a list a images and a numpy array (usually intrinsics matrix)
    """

    def __init__(self, root, train=True, sequence_length=3, transform=None, skip_frames=1, use_frame_index=False):
        """
        Args:
            root: dataset path
            train:
            sequence_length: number of images for training 训练要用的图片数
            transform: custom_transforms
            skip_frames: jump sampling from video  连续帧数
            use_frame_index: filter out static-camera frames in video
        """

        np.random.seed(0)
        random.seed(0)

        # 数据集路径
        self.root = Path(root)/'training'
        scene_list_path = self.root/'train.txt' if train else self.root/'val.txt'
        # train.txt 记录了图片名列表，folder 就是每个图片的文件名，将root拼接上每个folder

        # root/scene_n
        self.scenes = [self.root/folder[:-1]
                       for folder in open(scene_list_path)]

        self.transform = transform
        self.k = skip_frames
        self.use_frame_index = use_frame_index
        self.crawl_folders(sequence_length)  # 执行这个method，储存所有 { K, tgt_img, ref_img } 样本对的 self.samples 就准备好了

    # 爬取整个文件夹里的文件：
    def crawl_folders(self, sequence_length):
        # k skip frames
        sequence_set = []
        demi_length = (sequence_length-1)//2
        #
        shifts = list(range(-demi_length * self.k,
                      demi_length * self.k + 1, self.k))
        shifts.pop(demi_length)

        # scene = root/scene_n
        for scene in self.scenes:
            # numpy 获取 root/scene_n/cam.txt
            intrinsics = np.genfromtxt(
                scene/'cam.txt').astype(np.float32).reshape((3, 3))

            # ['root/scene_1/0000000.jpg', ...]
            imgs = sorted(scene.files('*.jpg'))

            if self.use_frame_index:
                assert (os.path.exists(scene/'frame_index.txt') == True)
                frame_index = [int(index)
                               for index in open(scene/'frame_index.txt')]
                imgs = [imgs[d] for d in frame_index]

            if len(imgs) < sequence_length:
                continue

            # i 大小：len(imgs) - 2 * demi_length
            for i in range(demi_length * self.k, len(imgs)-demi_length * self.k):
                # { K，tgt图片路径，ref图片路径 }
                sample = {'intrinsics': intrinsics,
                          'tgt': imgs[i], 'ref_imgs': []}

                # j 大小：2 * demi_length
                for j in shifts:
                    sample['ref_imgs'].append(imgs[i+j])

                # sequence_set 大小：len(imgs) - 2 * demi_length
                sequence_set.append(sample)

        self.samples = sequence_set

    def __getitem__(self, index):
        sample = self.samples[index]
        # 把路径读取成图片
        tgt_img = load_as_float(sample['tgt'])
        ref_imgs = [load_as_float(ref_img) for ref_img in sample['ref_imgs']]
        # 预处理
        if self.transform is not None:
            imgs, intrinsics = self.transform(
                [tgt_img] + ref_imgs, np.copy(sample['intrinsics'])
            )
            tgt_img = imgs[0]
            ref_imgs = imgs[1:]
        else:
            intrinsics = np.copy(sample['intrinsics'])
        return tgt_img, ref_imgs, intrinsics

    def __len__(self):
        return len(self.samples)
