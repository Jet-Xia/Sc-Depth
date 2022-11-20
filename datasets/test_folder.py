import torch.utils.data as data
import numpy as np
from imageio import imread
from path import Path
import torch
from scipy import sparse


def load_sparse_depth(filename):
    sparse_depth = sparse.load_npz(filename)
    depth = sparse_depth.todense()
    return np.array(depth)


def crawl_folder(folder, dataset='nyu'):

    imgs = sorted((folder/'color/').files('*.png'))
    if len(imgs) == 0:
        imgs = sorted((folder/'color/').files('*.jpg'))

    if dataset in ['nyu', 'tum', 'bonn']:
        depths = sorted((folder/'depth/').files('*.png'))
    elif dataset == 'kitti':
        depths = sorted((folder/'depth/').files('*.npy'))
    elif dataset == 'cs':
        depths = sorted((folder/'depth/').files('*.npy'))
    elif dataset == 'ddad':
        depths = sorted((folder/'depth/').files('*.npz'))

    return imgs, depths


class TestSet(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
        root/color/0000000.png
        root/depth/0000000.npz or png
    """

    def __init__(self, root, transform=None, dataset='nyu'):
        self.root = Path(root)/'testing'
        self.transform = transform
        self.dataset = dataset
        self.imgs, self.depths = crawl_folder(self.root, self.dataset)

    def __getitem__(self, index):
        img = imread(self.imgs[index]).astype(np.float32)

        if self.dataset in ['nyu']:
            depth = torch.from_numpy(
                imread(self.depths[index]).astype(np.float32)).float()/5000
        if self.dataset in ['bonn', 'tum']:
            depth = torch.from_numpy(
                imread(self.depths[index]).astype(np.float32)).float()/1000
        elif self.dataset == 'kitti':
            depth = torch.from_numpy(
                np.load(self.depths[index]).astype(np.float32))
        elif self.dataset == 'ddad':
            depth = torch.from_numpy(load_sparse_depth(
                self.depths[index]).astype(np.float32))
        elif self.dataset == 'cs':
            depth = torch.from_numpy(
                np.load(self.depths[index]).astype(np.float32))

        # crop cityscape images
        if self.dataset == 'cs':
            h = img.shape[0]
            img = img[0:int(h*0.75)]
            depth = depth[0:int(h*0.75)]

        if self.transform is not None:
            img, _ = self.transform([img], None)
            img = img[0]

        return img, depth

    def __len__(self):
        return len(self.imgs)
