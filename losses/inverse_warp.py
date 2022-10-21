from __future__ import division
import torch
import torch.nn.functional as F
from kornia.geometry.depth import depth_to_3d


# R
def euler2mat(angle):
    """Convert euler angles to rotation matrix.
     Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
    Args:
        angle: rotation angle along 3 axis (in radians) -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the euler angles -- size = [B, 3, 3]
    """
    B = angle.size(0)
    x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach()*0
    ones = zeros.detach()+1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz,  cosz, zeros,
                        zeros, zeros,  ones], dim=1).reshape(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros,  siny,
                        zeros,  ones, zeros,
                        -siny, zeros,  cosy], dim=1).reshape(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros,  cosx, -sinx,
                        zeros,  sinx,  cosx], dim=1).reshape(B, 3, 3)

    rotMat = xmat @ ymat @ zmat
    return rotMat


# R
def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: first three coeff of quaternion of rotation. fourht is then computed to have a norm of 1 -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = torch.cat([quat[:, :1].detach()*0 + 1, quat], dim=1)
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:,
                                            1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat


# [R | T]
def pose_vec2mat(vec, rotation_mode='euler'):
    """
    Convert 6DoF parameters to transformation matrix.
    Args:s
        vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
    Returns:
        A transformation matrix -- [B, 3, 4]
    """
    translation = vec[:, :3].unsqueeze(-1)  # [B, 3, 1]
    rot = vec[:, 3:]
    if rotation_mode == 'euler':
        rot_mat = euler2mat(rot)  # [B, 3, 3]
    elif rotation_mode == 'quat':
        rot_mat = quat2mat(rot)  # [B, 3, 3]
    transform_mat = torch.cat([rot_mat, translation], dim=2)  # [B, 3, 4]
    return transform_mat


def inverse_warp(img, depth, ref_depth, pose, intrinsics, padding_mode='zeros'):
    """
    Inverse warp a source image to the target image plane.
    Args:
        img: Is, the source image (where to sample pixels) -- [B, 3, H, W]
        depth: depth_t, depth map of the target image -- [B, 1, H, W]
        ref_depth: depth_s, the source depth map (where to sample depth) -- [B, 1, H, W]
        pose: 6DoF pose parameters from target to source -- [B, 6]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
    Returns:
        projected_img: Source image warped to the target image plane
        projected_depth: sampled depth from source image  
        computed_depth: computed depth of source image using the target depth
    """
    B, _, H, W = img.size()

    # 这一部分就是用 It 的 depth_t 与 T(t->s) 得到坐标变换矩阵 pix_coords
    # [R | T]
    T = pose_vec2mat(pose)  # [B,3,4]

    # M = K·[R | T]
    M = torch.matmul(intrinsics, T)[:, :3, :]   # [B,3,4]

    # 用 depth_to_3d 输入 depth 与内参就能还原三维坐标，会自动获取 depth 的 h，w，不需要自己创建像素坐标矩阵
    world_points = depth_to_3d(depth, intrinsics)   # [B, (x,y,d), H, W]   [B,3,H,W]

    # 把三维点的欧式坐标变换成齐次坐标
    world_points = torch.cat([world_points, torch.ones(B,1,H,W).type_as(img)], 1)   # [B, (x,y,z,1), H, W]   [B,4,H,W]

    # p = M·P = (m1·P, m2·P, m3·P)  得到目标二维点齐次坐标
    cam_points = torch.matmul(M, world_points.view(B, 4, -1))   # [B, (m1·P, m2·P, m3·P), HW]   [B,3,HW]

    # (m1·P, m2·P, m3·P) / m3·P = (m1·P/m3·P, m2·P/m3·P, 1) ->  (u, v)
    pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + 1e-7)
    pix_coords = pix_coords.view(B, 2, H, W)    # [B,2,HW] -> [B,2,H,W]
    pix_coords = pix_coords.permute(0, 2, 3, 1)  # [B, H, W, (u,v)]

    # 标准化到 [-1,1] 的区间，得到坐标变换矩阵，为了后面使用 F.grid_sample
    pix_coords[..., 0] /= W - 1     # u/(W-1)
    pix_coords[..., 1] /= H - 1     # v/(H-1)
    pix_coords = (pix_coords - 0.5) * 2

    # [B, m3·P, H, W]
    computed_depth = cam_points[:, 2, :].unsqueeze(1).view(B, 1, H, W)

    # Is == wraping ==> ^Is ~ It
    # 这里就是用双线性插值
    projected_img = F.grid_sample(img, pix_coords, padding_mode=padding_mode, align_corners=False)   # [B, 3, H, W]

    # 把 Is 的 depth 也 wrap 了,        depth_s ==> ^depth_s ~ depth_t
    projected_depth = F.grid_sample(ref_depth, pix_coords, padding_mode=padding_mode, align_corners=False)   # [B, 1, H, W]

    return projected_img, projected_depth, computed_depth


def inverse_rotation_warp(img, rot, intrinsics, padding_mode='zeros'):

    B, _, H, W = img.size()

    R = euler2mat(rot)  # [B, 3, 3]
    Mr = torch.matmul(intrinsics, R)

    # depth = 1
    world_points = depth_to_3d(torch.ones(B, 1, H, W).type_as(img), intrinsics)   # B 3 H W
    cam_points = torch.matmul(Mr, world_points.view(B, 3, -1))

    pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + 1e-7)
    pix_coords = pix_coords.view(B, 2, H, W)
    pix_coords = pix_coords.permute(0, 2, 3, 1)
    pix_coords[..., 0] /= W - 1
    pix_coords[..., 1] /= H - 1
    pix_coords = (pix_coords - 0.5) * 2  

    projected_img = F.grid_sample(img, pix_coords, padding_mode=padding_mode, align_corners=True)

    return projected_img
