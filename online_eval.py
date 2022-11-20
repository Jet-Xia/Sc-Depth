import time
from config import get_opts, get_training_size
import onnxruntime
import numpy as np
import cv2 as cv


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


hparams = get_opts()
img_size = get_training_size(hparams.dataset_name)
counter = 0
start_time = time.time()

cap = cv.VideoCapture(0)  # 调用摄像头‘0’一般是打开电脑自带摄像头，‘1’是打开外部摄像头（只有一个摄像头的情况）
# width, height = img_size[1], img_size[0]
# cap.set(cv.CAP_PROP_FRAME_WIDTH, width)  # 设置图像宽度
# cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)  # 设置图像高度

# 显示图像
while True:
    ret, frame = cap.read()  # 读取图像(frame就是读取的视频帧，对frame处理就是对整个视频的处理)
    # print(ret)

    img = cv.resize(frame, (img_size[1], img_size[0]), interpolation=cv.INTER_LINEAR).astype(np.float32)
    img = np.transpose(img, (2, 0, 1))  # h,w,c -> c,h,w
    img = img / 225
    img = np.expand_dims(img, axis=0)  # c,h,w -> b,c,h,w

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

    vis = visualize_depth(pred_depth[0])
    # 和 cv.imwrite 输出0-255分布是反的
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