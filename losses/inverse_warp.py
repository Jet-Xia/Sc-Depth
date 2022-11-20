from __future__ import division
import torch
import torch.nn.functional as F
from kornia.geometry.depth import depth_to_3d
from kornia.geometry.conversions import angle_axis_to_rotation_matrix


def pose_vec2mat(vec):
    """
    Convert 6DoF parameters to transformation matrix.
    Args:s
        vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
    Returns:
        A transformation matrix -- [B, 3, 4]
    """
    translation = vec[:, :3].unsqueeze(-1)  # [B, 3, 1]
    rot_mat = angle_axis_to_rotation_matrix(vec[:, 3:])
    transform_mat = torch.cat([rot_mat, translation], dim=2)  # [B, 3, 4]
    return transform_mat


def inverse_warp(img, depth, ref_depth, pose, intrinsics, padding_mode='zeros'):
    """
    Inverse warp a source image to the target image plane.
    Args:
        img: the source image (where to sample pixels) -- [B, 3, H, W]
        depth: depth map of the target image -- [B, 1, H, W]
        ref_depth: the source depth map (where to sample depth) -- [B, 1, H, W] 
        pose: 6DoF pose parameters from target to source -- [B, 6]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
    Returns:
        projected_img: Source image warped to the target image plane
        projected_depth: sampled depth from source image  
        computed_depth: computed depth of source image using the target depth
    """
    B, _, H, W = img.size()

    T = pose_vec2mat(pose)  # [B,3,4]
    P = torch.matmul(intrinsics, T)[:, :3, :]

    world_points = depth_to_3d(depth, intrinsics)  # B 3 H W
    world_points = torch.cat(
        [world_points, torch.ones(B, 1, H, W).type_as(img)], 1)
    cam_points = torch.matmul(P, world_points.view(B, 4, -1))

    pix_coords = cam_points[:, :2, :] / \
        (cam_points[:, 2, :].unsqueeze(1) + 1e-7)
    pix_coords = pix_coords.view(B, 2, H, W)
    pix_coords = pix_coords.permute(0, 2, 3, 1)
    pix_coords[..., 0] /= W - 1
    pix_coords[..., 1] /= H - 1
    pix_coords = (pix_coords - 0.5) * 2

    computed_depth = cam_points[:, 2, :].unsqueeze(1).view(B, 1, H, W)

    projected_img = F.grid_sample(
        img, pix_coords, padding_mode=padding_mode, align_corners=False)
    projected_depth = F.grid_sample(
        ref_depth, pix_coords, padding_mode=padding_mode, align_corners=False)

    return projected_img, projected_depth, computed_depth


def inverse_rotation_warp(img, rot, intrinsics, padding_mode='zeros'):

    B, _, H, W = img.size()

    R = angle_axis_to_rotation_matrix(rot)  # [B, 3, 3]
    P = torch.matmul(intrinsics, R)

    world_points = depth_to_3d(torch.ones(
        B, 1, H, W).type_as(img), intrinsics)  # B 3 H W
    cam_points = torch.matmul(P, world_points.view(B, 3, -1))

    pix_coords = cam_points[:, :2, :] / \
        (cam_points[:, 2, :].unsqueeze(1) + 1e-7)
    pix_coords = pix_coords.view(B, 2, H, W)
    pix_coords = pix_coords.permute(0, 2, 3, 1)
    pix_coords[..., 0] /= W - 1
    pix_coords[..., 1] /= H - 1
    pix_coords = (pix_coords - 0.5) * 2

    projected_img = F.grid_sample(
        img, pix_coords, padding_mode=padding_mode, align_corners=False)

    return projected_img
