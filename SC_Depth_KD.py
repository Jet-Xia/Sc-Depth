import numpy
from pytorch_lightning import LightningModule
import losses.loss_functions as LossF
from models.KDDepthNet import KDDepthNet
from models.KDPoseNet import KDPoseNet
from visualization import *
from path import Path


class KD_SC_Depth(LightningModule):
    def __init__(self, hparams):
        super(KD_SC_Depth, self).__init__()
        self.save_hyperparameters()

        # model
        # DepthNet(net='resnet', num_layers=18, pretrained=True)
        self.depth_net = KDDepthNet()
        self.pose_net = KDPoseNet()

    def configure_optimizers(self):
        optim_params = [
            {'params': self.depth_net.parameters(), 'lr': self.hparams.hparams.lr},
            {'params': self.pose_net.parameters(), 'lr': self.hparams.hparams.lr}
        ]
        optimizer = torch.optim.Adam(optim_params)
        return [optimizer]

    def training_step(self, batch, batch_idx):
        tgt_img, teacher_depth, ref_imgs, intrinsics = batch

        # network forward
        # { tgt_img 编号: d+i ;  ref_imgs 编号: (0+i-d开始 , 长度为 2d); tgt个数：len-2d}
        # 对长度为 2d 的 ref_img 每一个都预测它的 depth
        tgt_depth = self.depth_net(tgt_img)
        ref_depths = [self.depth_net(im) for im in ref_imgs]

        # 对长度为 2d 的 ref_img 每一个都与 tgt_img 预测一个 pose 和反向 pose_inv
        poses = [self.pose_net(tgt_img, im) for im in ref_imgs]
        poses_inv = [self.pose_net(im, tgt_img) for im in ref_imgs]

        # compute loss
        # α，γ，β
        w1 = self.hparams.hparams.photo_weight
        w2 = self.hparams.hparams.geometry_weight
        w3 = self.hparams.hparams.smooth_weight
        w4 = self.hparams.hparams.mask_rank_weight

        # 把预测出的 tgt 和 ref 的原图、depth、pose传进loss
        loss_1, loss_2, dynamic_mask = LossF.photo_and_geometry_loss(tgt_img, ref_imgs, tgt_depth, ref_depths,
                                                                     intrinsics, poses, poses_inv, self.hparams.hparams)
        # smooth loss
        loss_3 = LossF.compute_smooth_loss(tgt_depth, tgt_img)

        # teacher model loss
        loss_4 = LossF.mask_ranking_loss(tgt_depth, teacher_depth, dynamic_mask)

        # L = α·Lp + γ·Lg + β·Ls
        loss = w1 * loss_1 + w2 * loss_2 + w3 * loss_3 + w4 * loss_4

        # b, c, h, w = dynamic_mask.shape
        # name = Path('F:/dataset/output/ms') / '{} ms.jpg'.format(batch_idx)
        # cv2.imwrite(name, (dynamic_mask.squeeze(0).view(h,w,c) * 255).detach().cpu().numpy().astype(np.uint16))
        # name = Path('F:/dataset/output/ms') / '{} tgt.jpg'.format(batch_idx)
        # cv2.imwrite(name, cv2.cvtColor(tgt_img.squeeze(0).view(h,w,3).detach().cpu().numpy().astype(np.uint16), cv2.COLOR_RGB2BGR))
        # i = 0
        # for ref_img in ref_imgs:
        #     i = i + 1
        #     name = Path('F:/dataset/output/ms') / '{} ref {}.jpg'.format(batch_idx, i)
        #     cv2.imwrite(name, cv2.cvtColor(ref_img.squeeze(0).view(h,w,3).detach().cpu().numpy().astype(np.uint16), cv2.COLOR_RGB2BGR))

        # create logs
        self.log('train/total_loss', loss)
        self.log('train/photo_loss', loss_1)
        self.log('train/geometry_loss', loss_2)
        self.log('train/smooth_loss', loss_3)
        self.log('train/mask_ranking_loss', loss_4)

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
            tgt_depth = self.depth_net(tgt_img)
            ref_depths = [self.depth_net(im) for im in ref_imgs]
            poses = [self.pose_net(tgt_img, im) for im in ref_imgs]
            poses_inv = [self.pose_net(im, tgt_img) for im in ref_imgs]
            loss_1, loss_2 = LossF.photo_and_geometry_loss(tgt_img, ref_imgs, tgt_depth, ref_depths,
                                                           intrinsics, poses, poses_inv)
            errs = {'photo_loss': loss_1.item(), 'geometry_loss': loss_2.item()}
        else:
            print('wrong validation mode')

        if self.global_step < 10:
            return errs

        # plot
        if batch_idx < 3:
            vis_img = visualize_image(tgt_img[0])  # (3, H, W)
            vis_depth = visualize_depth(tgt_depth[0, 0])  # (3, H, W)
            stack = torch.cat([vis_img, vis_depth], dim=1).unsqueeze(0)  # (3, 2*H, W)
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
