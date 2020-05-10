import cv2
import numpy as np

def IoU(a,b):
    top_l_x = max(a[0], b[0])
    top_l_y = max(a[1], b[1])
    low_r_x = min(a[2],b[2])
    low_r_y = min(a[3], b[3])


    RoL = np.array((top_l_x,top_l_y,low_r_x,low_r_y))
    print(RoL)

    iou_h = RoL[3]-RoL[1]
    iou_w = RoL[2]-RoL[0]

    if iou_h <= 0 or iou_w <= 0:
        return 0.0

    s_RoL = iou_h * iou_w


    s_a = (a[3]-a[1]) * (a[2] - a[0])
    s_b = (b[3]-b[1]) * (b[2] - b[0])


    iou = abs(s_RoL) / abs(s_a + s_b - s_RoL)

    return iou

if __name__ == "__main__":
    a = np.array((50, 50, 150, 150), dtype = np.float32)
    b = np.array((60, 60, 170, 160), dtype = np.float32)
    s = IoU(a,b)
    print(s)
