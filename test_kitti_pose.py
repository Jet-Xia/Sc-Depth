import numpy as np
import torch
from imageio import imread
from path import Path
from tqdm import tqdm

import datasets.custom_transforms as custom_transforms
from config import get_opts, get_training_size
from losses.inverse_warp import pose_vec2mat
from SC_Depth import SC_Depth


class kitti_pose_test_framework(object):
    def __init__(self, root, sequence, seq_length=3, step=1):
        self.root = root
        self.img_files, self.poses, self.sample_indices = read_kitti_scene_data(
            self.root, sequence, seq_length, step)

    def generator(self):
        for img_list, pose_list, sample_list in zip(self.img_files, self.poses, self.sample_indices):
            for snippet_indices in sample_list:
                imgs = [imread(img_list[i]).astype(np.float32)
                        for i in snippet_indices]

                poses = np.stack([pose_list[i] for i in snippet_indices])
                first_pose = poses[0]
                poses[:, :, -1] -= first_pose[:, -1]
                compensated_poses = np.linalg.inv(first_pose[:, :3]) @ poses

                yield {'imgs': imgs,
                       'poses': compensated_poses
                       }

    def __iter__(self):
        return self.generator()

    def __len__(self):
        return sum(len(imgs) for imgs in self.img_files)


def read_kitti_scene_data(data_root, sequence, seq_length=5, step=1):
    data_root = Path(data_root)
    im_sequences = []
    poses_sequences = []
    indices_sequences = []
    demi_length = (seq_length - 1) // 2
    shift_range = np.array(
        [step*i for i in range(-demi_length, demi_length + 1)]).reshape(1, -1)

    print('getting test metadata for the sequence : {}'.format(sequence))

    poses = np.genfromtxt(
        data_root/'testing/poses/{}.txt'.format(sequence)).astype(np.float64).reshape(-1, 3, 4)
    imgs = sorted(
        (data_root/'testing/sequences/{}'.format(sequence)/'image_2').files('*.png'))
    # construct 5-snippet sequences
    tgt_indices = np.arange(demi_length, len(
        imgs) - demi_length).reshape(-1, 1)
    snippet_indices = shift_range + tgt_indices
    im_sequences.append(imgs)
    poses_sequences.append(poses)
    indices_sequences.append(snippet_indices)

    return im_sequences, poses_sequences, indices_sequences


@torch.no_grad()
def test_sequence(pose_net, dataset_dir, sequence='09', seq_length=5):

    test_transform = custom_transforms.Compose([
        custom_transforms.RescaleTo(get_training_size("kitti")),
        custom_transforms.ArrayToTensor(),
        custom_transforms.Normalize()]
    )

    framework = kitti_pose_test_framework(
        dataset_dir, sequence=sequence, seq_length=seq_length)

    print('{} snippets to test'.format(len(framework)))
    errors = np.zeros((len(framework), 2), np.float32)

    for j, sample in enumerate(tqdm(framework)):
        imgs = sample['imgs']

        squence_imgs, _ = test_transform(imgs, None)
        squence_imgs = [im.unsqueeze(0).cuda() for im in squence_imgs]

        global_pose = np.eye(4)
        poses = []
        poses.append(global_pose[0:3, :])

        for iter in range(seq_length - 1):
            pose = pose_net(squence_imgs[iter], squence_imgs[iter+1])
            pose_mat = pose_vec2mat(pose).squeeze(0).cpu().numpy()

            pose_mat = np.vstack([pose_mat, np.array([0, 0, 0, 1])])
            global_pose = global_pose @  np.linalg.inv(pose_mat)
            poses.append(global_pose[0:3, :])

        final_poses = np.stack(poses, axis=0)

        ATE, RE = compute_pose_error(sample['poses'], final_poses)
        errors[j] = ATE, RE

    mean_errors = errors.mean(0)
    std_errors = errors.std(0)
    error_names = ['ATE', 'RE']
    print('')
    print("Results for sequence {}".format(sequence))
    print("\t {:>10}, {:>10}".format(*error_names))
    print("mean \t {:10.4f}, {:10.4f}".format(*mean_errors))
    print("std \t {:10.4f}, {:10.4f}".format(*std_errors))


def compute_pose_error(gt, pred):
    RE = 0
    snippet_length = gt.shape[0]
    scale_factor = np.sum(gt[:, :, -1] * pred[:, :, -1]
                          )/np.sum(pred[:, :, -1] ** 2)
    ATE = np.linalg.norm(
        (gt[:, :, -1] - scale_factor * pred[:, :, -1]).reshape(-1))
    for gt_pose, pred_pose in zip(gt, pred):
        # Residual matrix to which we compute angle's sin and cos
        R = gt_pose[:, :3] @ np.linalg.inv(pred_pose[:, :3])
        s = np.linalg.norm([R[0, 1]-R[1, 0],
                            R[1, 2]-R[2, 1],
                            R[0, 2]-R[2, 0]])
        c = np.trace(R) - 1
        # Note: we actually compute double of cos and sin, but arctan2 is invariant to scale
        RE += np.arctan2(s, c)

    return ATE/snippet_length, RE/snippet_length


@torch.no_grad()
def main():
    hparams = get_opts()

    # initialize network
    system = SC_Depth(hparams)
    system = system.load_from_checkpoint(hparams.ckpt_path, strict=False)

    pose_net = system.pose_net
    pose_net.cuda()
    pose_net.eval()

    test_sequence(pose_net, hparams.dataset_dir, sequence='09')
    test_sequence(pose_net, hparams.dataset_dir, sequence='10')


if __name__ == '__main__':
    main()
