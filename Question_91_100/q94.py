import cv2
import numpy as np

np.random.seed(0)

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


def Cropping(img, gt, K = 200,L= 60, th = 0.5):
    H, W, C = img.shape
    out = img.copy()

    for i in range(K):
        x1 = np.random.randint(W-L)
        y1 = np.random.randint(H-L)
        x2 = x1 + L
        y2 = y1 + L

        crop = np.array((x1,y1,x2,y2),dtype = np.float32)

        iou = IoU(gt,crop)

        if iou >= th:
            cv2.rectangle(out, (x1,y1),(x2, y2),(0,0,255), 1)
            label = 1

        elif iou < th:
            cv2.rectangle(out, (x1,y1),(x2, y2),(255,0,0), 1)
            label = 0

    return out.astype(np.uint8)

img = cv2.imread("./image_91_100/imori_1.jpg").astype(np.float32)
gt = np.array((47, 41, 129, 103), dtype = np.float32)
out = Cropping(img, gt, 200, 60, 0.5)
cv2.rectangle(out, (gt[0],gt[1]),(gt[2], gt[3]),(0,255,0), 1)

cv2.imwrite("./image_91_100/answer_94.jpg", out)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
