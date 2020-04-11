import cv2
import numpy as np

def quantization(_img):
    img = _img.copy()
    out = np.zeros_like(_img, dtype = np.float32)

    for i in range(4):
        #青
        out[:,:,0][(i*64. <= img[:,:,0]) & (img[:,:,0] < (i+1) * 64.)] = 32. * (i+1)

        #緑
        out[:,:,1][np.where( (i*64. <= img[:,:,1]) & (img[:,:,1] < (i+1) * 64.))] = 32. * (i+1)

        #赤
        out[:,:,2][np.where( (i*64. <= img[:,:,2]) & (img[:,:,2] < (i+1) * 64.))] = 32. * (i+1)

    out = out.astype(np.uint8)
    return out


img = cv2.imread("./image_01_10/imori.jpg").astype(np.float32)

img_ans= quantization(img)

cv2.imwrite("./image_01_10/answer6.jpg", img_ans)
cv2.imshow("result", img_ans)
cv2.waitKey(0)
cv2.destroyAllWindows()
