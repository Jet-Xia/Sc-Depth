import configargparse


def get_opts():
    parser = configargparse.ArgumentParser()

    # configure file
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')

    # dataset
    parser.add_argument('--dataset_dir', type=str)
    parser.add_argument('--dataset_name', type=str, required=True, default='kitti',
                        choices=['kitti', 'nyu',  'tum', 'bonn', 'ddad'])
    parser.add_argument('--sequence_length', type=int,
                        default=3, help='number of images for training')
    parser.add_argument('--skip_frames', type=int, default=1,
                        help='skip frames from video')
    parser.add_argument('--with_frame_index', action='store_true',
                        help='a file that contains index of frames for training')

    # model
    parser.add_argument('--model_version', type=str,
                        default='v1', choices=['v1', 'v2', 'v2p', 'v3', 'v3p'])
    parser.add_argument('--resnet_layers', type=int, default=18)
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='pretrained checkpoint path to load')
    parser.add_argument('--KD', type=str, default='no')

    # loss
    parser.add_argument('--photo_weight', type=float,
                        default=1.0, help='photometric loss weight')
    parser.add_argument('--geometry_weight', type=float,
                        default=0.1, help='geometry loss weight')
    parser.add_argument('--smooth_weight', type=float,
                        default=0.1, help='smoothness loss weight')

    # for v2
    parser.add_argument('--rot_t_weight', type=float,
                        default=1.0, help='rotation triplet loss weight')
    parser.add_argument('--rot_c_weight', type=float,
                        default=1.0, help='rotation consistency loss weight')

    # for v3
    parser.add_argument('--mask_rank_weight', type=float, default=0.1,
                        help='ranking loss with dynamic mask sampling')
    parser.add_argument('--normal_consistency_weight', type=float,
                        default=0.1, help='edge-guided sampling for normal ranking loss')
    parser.add_argument('--normal_rank_weight', type=float, default=0.1,
                        help='edge-guided sampling for normal ranking loss')

    parser.add_argument('--val_mode', type=str, default='depth',
                        choices=['photo', 'depth'], help='how to run validation')

    # for kd
    parser.add_argument('--kd_weight', type=float,
                        default=0.3, help='kd loss weight')

    # for ablation study
    parser.add_argument('--no_ssim', action='store_true',
                        help='use ssim in photometric loss')
    parser.add_argument('--no_auto_mask', action='store_true',
                        help='masking invalid static points')
    parser.add_argument('--no_dynamic_mask',
                        action='store_true', help='masking dynamic regions')
    parser.add_argument('--no_min_optimize', action='store_true',
                        help='optimize the minimum loss')

    # training options
    parser.add_argument('--exp_name', type=str, help='experiment name')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--epoch_size', type=int,
                        default=1000, help='number of training steps per epoch')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')

    # inference options
    parser.add_argument('--input_dir', type=str, help='input image path')
    parser.add_argument('--output_dir', type=str, help='output depth path')
    parser.add_argument('--save-vis', action='store_true',
                        help='save depth visualization')
    parser.add_argument('--save-depth', action='store_true',
                        help='save depth with factor 1000')

    return parser.parse_args()


def get_training_size(dataset_name):

    if dataset_name == 'kitti':
        training_size = [256, 832]
    elif dataset_name == 'ddad':
        training_size = [384, 640]
    elif dataset_name == 'nyu':
        training_size = [256, 320]
    elif dataset_name == 'bonn':
        training_size = [256, 320]
    elif dataset_name == 'tum':
        training_size = [256, 320]
    else:
        print('unknown dataset type')

    return training_size
