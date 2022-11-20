import time
from config import get_opts, get_training_size
import torchvision.transforms as transforms
import torch
import numpy as np
import cv2 as cv
from SC_Depth_KD import KD_SC_Depth
from visualization import *

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


def scale_torch(img):
    """
    Scale the image and output it in torch.tensor.
    :param img: input rgb is in shape [H, W, C], input depth/disp is in shape [H, W]
    :param scale: the scale factor. float
    :return: img. [C, H, W]
    """
    if len(img.shape) == 2:
        img = img[np.newaxis, :, :]
    if img.shape[2] == 3:
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225))])
        img = transform(img)
    else:
        img = img.astype(np.float32)
        img = torch.from_numpy(img)
    return img


@torch.no_grad()
def inference(depth_model, rgb):
    input = rgb.to(device)
    depth = depth_model(input)
    return depth


hparams = get_opts()
img_size = get_training_size(hparams.dataset_name)
counter = 0
start_time = time.time()

cap = cv.VideoCapture(0)  # 调用摄像头‘0’一般是打开电脑自带摄像头，‘1’是打开外部摄像头（只有一个摄像头的情况）
width, height = img_size[1], img_size[0]
# cap.set(cv.CAP_PROP_FRAME_WIDTH, width)  # 设置图像宽度
# cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)  # 设置图像高度

system = KD_SC_Depth(hparams)
system = system.load_from_checkpoint(hparams.ckpt_path, strict=False)

depth_model = system.depth_net
depth_model.to(device)
depth_model.eval()

# 显示图像
while True:
    ret, frame = cap.read()  # 读取图像(frame就是读取的视频帧，对frame处理就是对整个视频的处理)
    # print(ret)

    rgb = frame[:, :, ::-1].copy()

    A_resize = cv2.resize(rgb, (width, height))
    img_torch = scale_torch(A_resize)[None, :, :, :]

    pred_depth = inference(depth_model, img_torch)

    vis = visualize_depth(pred_depth[0, 0]).permute(1, 2, 0).numpy() * 255
    vis_c = vis[:, :, ::-1].copy()

    counter += 1  # 计算帧数
    if (time.time() - start_time) != 0:  # 实时显示帧数
        cv.putText(frame, "FPS {0}".format(float('%.1f' % (counter / (time.time() - start_time)))), (30, 50),
                    cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv.putText(frame, "Latency Time {0} s".format(float('%.1f' % (time.time() - start_time))), (30, 75),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv.imshow("frame", frame)
        cv.imshow("depth", vis_c.astype(np.uint8))
        # print("FPS: ", counter / (time.time() - start_time))
        counter = 0
        start_time = time.time()

    input_str = cv.waitKey(20)
    if input_str == ord('q'):  # 如过输入的是q就break，结束图像显示，鼠标点击视频画面输入字符
        break

cap.release()  # 释放摄像头
cv.destroyAllWindows()  # 销毁窗口