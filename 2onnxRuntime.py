import onnx
import onnxruntime
from config import get_opts, get_training_size
import numpy as np
import cv2 as cv
import torch
from imageio.v2 import imread, imwrite

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


def visualize_depth(depth, cmap=cv.COLORMAP_JET):
    """
    depth: (H, W)
    """
    x = depth
    x = np.nan_to_num(x)  # change nan to 0
    mi = np.min(x)  # get minimum depth
    ma = np.max(x)
    x = (x - mi) / (ma - mi + 1e-8)  # normalize to 0~1
    x = (255 * x).astype(np.uint8)
    x = cv.applyColorMap(x, cmap)
    return x

@torch.no_grad()
def main():
    hparams = get_opts()

    onnx_model = onnx.load("srcnn.onnx")
    try:
        onnx.checker.check_model(onnx_model)
    except Exception:
        print("Model incorrect")
    else:
        print("Model correct")

    img = imread('F:\\dataset\\ddad\\testing\\color\\000000.jpg').astype(np.float32)

    img_size = get_training_size(hparams.dataset_name)
    img = cv.resize(img, (img_size[1], img_size[0]), interpolation=cv.INTER_LINEAR).astype(np.float32)
    img = np.transpose(img, (2, 0, 1))    # h,w,c -> c,h,w
    img = img / 225
    img = np.expand_dims(img, axis=0)     # c,h,w -> b,c,h,w

    # Normalization
    mean = 0.45
    std = 0.225
    img = img - mean
    img = img / std

    # onnx processing
    # 虽然输入是 ndarray， 但是不用BGR2RGB，变成 b,c,h,w 后直接输入模型就行
    ort_session = onnxruntime.InferenceSession("srcnn.onnx")
    ort_inputs = {'image': img}
    ort_output = ort_session.run(['depth'], ort_inputs)[0]

    # output
    pred_depth = np.squeeze(ort_output, 0)

    if hparams.save_vis:
        vis = visualize_depth(pred_depth[0])
        # 和 cv.imwrite 输出0-255分布是反的
        imwrite('{}.jpg'.format(0), vis.astype(np.uint8))

    if hparams.save_depth:
        depth = pred_depth[0]
        np.save('{}.npy'.format(0), depth)


if __name__ == '__main__':
    main()