import numpy as np
import torch
# pytorch-lightning
from pytorch_lightning import LightningModule

import losses.loss_functions as LossF
from losses.inverse_warp import inverse_rotation_warp
from models.DepthNet import DepthNet
from models.PoseNet import PoseNet
from models.RectifyNet import RectifyNet
from visualization import *


class SC_DepthV2p(LightningModule):
    def __init__(self, hparams):
        super(SC_DepthV2p, self).__init__()
        self.save_hyperparameters()

        # model
        self.depth_net = DepthNet(self.hparams.hparams.resnet_layers)
        self.pose_net = PoseNet()
        self.rectify_net = RectifyNet()

    def rectify_imgs(self, tgt_img, ref_imgs, intrinsics):
        
        # use arn to pre-warp ref images
        rot1_list = [] 
        rot2_list = []
        rot3_list = []
        rot_augment_list = []
        rot_pred_list = []
        ref_imgs_warped = []
        for ref_img in ref_imgs:
            rot1 = self.rectify_net(tgt_img, ref_img)
            rot_warped_img = inverse_rotation_warp(ref_img, rot1, intrinsics)
            rot1_list += [rot1]

            rot2 = self.rectify_net(tgt_img, rot_warped_img)
            rot2_list += [rot2]

            rot3 = self.rectify_net(rot_warped_img, ref_img)
            rot3_list += [rot3]
            
            # it ranges [-0.05, +0.05]
            rot_augment = (torch.rand_like(rot1) - 0.5) / 10
            rot_pred = self.rectify_net(
                inverse_rotation_warp(
                    ref_img, 
                    rot_augment, 
                    intrinsics
                ),
                ref_img
                )

            ref_imgs_warped.append(rot_warped_img)
            rot_augment_list.append(rot_augment)
            rot_pred_list.append(rot_pred)

        # stack
        rot1 = torch.stack(rot1_list)
        rot2 = torch.stack(rot2_list)
        rot3 = torch.stack(rot3_list)
        rot_augment = torch.stack(rot_augment_list)
        rot_pred = torch.stack(rot_pred_list)

        # rot_supervised, rot1 is gt rotation for rot3
        loss_rot_supervised = (rot3 - rot1.detach()).abs().mean()

        # rot_2 should be 0
        loss_rot_after = rot2.abs().mean()

        # rot_augment loss, rot_augment is gt rotation for rot_pred
        loss_rot_augment = (rot_pred - rot_augment).abs().mean()
        
        return ref_imgs_warped, loss_rot_supervised, loss_rot_after, loss_rot_augment, rot1.abs().mean()

    def configure_optimizers(self):
        optim_params = [
            {'params': self.depth_net.parameters(), 'lr': self.hparams.hparams.lr},
            {'params': self.pose_net.parameters(), 'lr': self.hparams.hparams.lr},
            {'params': self.rectify_net.parameters(), 'lr': self.hparams.hparams.lr}
        ]
        optimizer = torch.optim.Adam(optim_params)

        return [optimizer]

    def training_step(self, batch, batch_idx):
        tgt_img, ref_imgs, intrinsics = batch

        # network forward
        tgt_depth = self.depth_net(tgt_img)
        ref_imgs_warped, loss_rs, loss_after, loss_augment, rot_before = self.rectify_imgs(tgt_img, ref_imgs, intrinsics)
        ref_depths = [self.depth_net(im) for im in ref_imgs_warped]
        poses = [self.pose_net(tgt_img, im) for im in ref_imgs_warped]
        poses_inv = [self.pose_net(im, tgt_img) for im in ref_imgs_warped]

        # compute loss
        w1 = self.hparams.hparams.photo_weight
        w2 = self.hparams.hparams.geometry_weight
        w3 = self.hparams.hparams.smooth_weight

        # w4 = self.hparams.hparams.rot_s_weight
        # w5 = self.hparams.hparams.rot_t_weight
        w4 = 0.1
        w5 = 0.1
        w6 = 1.0

        loss_1, loss_2 = LossF.photo_and_geometry_loss(tgt_img, ref_imgs_warped, tgt_depth, ref_depths,
                                                    intrinsics, poses, poses_inv, self.hparams.hparams)
        loss_3 = LossF.compute_smooth_loss(tgt_depth, tgt_img)

        loss = w1*loss_1 + w2*loss_2 + w3*loss_3 + w4*loss_rs + w5*loss_after + w6*loss_augment
        
        # create logs
        self.log('train/total_loss', loss)
        self.log('train/photo_loss', loss_1)
        self.log('train/geometry_loss', loss_2)
        self.log('train/smooth_loss', loss_3)
        self.log('train/rot_supervised_loss', loss_rs)
        self.log('train/rot_augment_loss', loss_augment)
        self.log('train/rot_before', rot_before)
        self.log('train/rot_after', loss_after)


        # add visualization for rectification
        if self.global_step % 100 == 0:
            vis_img = visualize_image(tgt_img[0]) # (3, H, W)
            vis_ref = visualize_image(ref_imgs[0][0]) # (3, H, W)
            vis_warp = visualize_image(ref_imgs_warped[0][0]) # (3, H, W)
            vis_all = torch.cat([vis_img, vis_ref, vis_warp], dim=1).unsqueeze(0) # (1, 3, 3*H, W)
            self.logger.experiment.add_images('train/tgt_ref_warp', vis_all, self.global_step)
        
        return loss
        
    def validation_step(self, batch, batch_idx):

        if self.hparams.hparams.val_mode == 'depth':
            tgt_img, gt_depth = batch
            tgt_depth = self.depth_net(tgt_img)
            errs = LossF.compute_errors(gt_depth, tgt_depth, self.hparams.hparams.dataset_name)
            
            errs = {'abs_diff': errs[0], 'abs_rel': errs[1],
                    'a1': errs[6], 'a2': errs[7], 'a3': errs[8]}

        elif self.hparams.hparams.val_mode == 'photo':
            tgt_img, ref_imgs, intrinsics = batch
            # tgt_depth, ref_depths, poses, poses_inv = self(tgt_img, ref_imgs)
            # loss_1, loss_2 = photo_and_geometry_loss(tgt_img, ref_imgs, tgt_depth, ref_depths,
            #                                             intrinsics, poses, poses_inv)
            # errs = {'photo_loss': loss_1.item(), 'geometry_loss': loss_2.item()}
        else:
            print('wrong validation mode')
   
        if self.global_step < 10:
            return errs

        # plot 
        if batch_idx < 3:
            vis_img = visualize_image(tgt_img[0]) # (3, H, W)
            vis_depth = visualize_depth(tgt_depth[0,0]) # (3, H, W)
            stack = torch.cat([vis_img, vis_depth], dim=1).unsqueeze(0) # (3, 2*H, W)
            self.logger.experiment.add_images('val/img_depth_{}'.format(batch_idx), stack, self.current_epoch)
        
        return errs

    def validation_epoch_end(self, outputs):

        if self.hparams.hparams.val_mode == 'depth':
            mean_rel = np.array([x['abs_rel'] for x in outputs]).mean()
            mean_diff = np.array([x['abs_diff'] for x in outputs]).mean()
            mean_a1 = np.array([x['a1'] for x in outputs]).mean()
            mean_a2 = np.array([x['a2'] for x in outputs]).mean()
            mean_a3 = np.array([x['a3'] for x in outputs]).mean()
            
            self.log('val_loss', mean_rel, prog_bar=True)
            self.log('val/abs_diff', mean_diff)
            self.log('val/abs_rel', mean_rel)
            self.log('val/a1', mean_a1, on_epoch=True)
            self.log('val/a2', mean_a2, on_epoch=True)
            self.log('val/a3', mean_a3, on_epoch=True)

        elif self.hparams.hparams.val_mode == 'photo':
            mean_pl = np.array([x['photo_loss'] for x in outputs]).mean()
            self.log('val_loss', mean_pl, prog_bar=True)
  
 