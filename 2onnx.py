import torch
from SC_Depth_KD import KD_SC_Depth
import cv2 as cv
import datasets.custom_transforms as custom_transforms
from config import get_opts, get_training_size
from SC_Depth import SC_Depth
from SC_DepthV2 import SC_DepthV2
from SC_DepthV3 import SC_DepthV3
from SC_DepthV3p import SC_DepthV3p
from visualization import *

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


@torch.no_grad()
def main():
    hparams = get_opts()

    if hparams.model_version == 'v1':
        system = SC_Depth(hparams)
    elif hparams.model_version == 'v2':
        system = SC_DepthV2(hparams)
    elif hparams.model_version == 'v3':
        system = SC_DepthV3(hparams)
    elif hparams.model_version == 'v3p':
        if hparams.KD == 'no':
            system = SC_DepthV3p(hparams)
        elif hparams.KD == 'yes':
            system = KD_SC_Depth(hparams)
    system = system.load_from_checkpoint(hparams.ckpt_path, strict=False)
    model = system.depth_net
    # model.to(device)
    model.eval()

    # training size
    training_size = get_training_size(hparams.dataset_name)

    # normaliazation
    inference_transform = custom_transforms.Compose([
        custom_transforms.RescaleTo(training_size),
        custom_transforms.ArrayToTensor(),
        custom_transforms.Normalize()]
    )

    depth_model = system.depth_net

    img_size = get_training_size(hparams.dataset_name)
    tensor_img = torch.randn(1, 3, img_size[0], img_size[1])
    # tensor_img.to(device)

    torch.onnx.export(
        depth_model,
        tensor_img,
        "srcnn.onnx",
        opset_version=11,
        input_names=['image'],
        output_names=['depth'])


if __name__ == '__main__':
    main()