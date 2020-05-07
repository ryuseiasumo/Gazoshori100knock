import cv2
import numpy as np

# Grayscale
def BGR2GRAY(img):
    # Grayscale
    gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
    return gray


gabor_0 = cv2.getGaborKernel((11, 11), 1.5, np.radians(0), 3, 1.2 ,0 ,cv2.CV_64F)
gabor_45 = cv2.getGaborKernel((11, 11), 1.5, np.radians(45), 3, 1.2 ,0 ,cv2.CV_64F)
gabor_90 = cv2.getGaborKernel((11, 11), 1.5, np.radians(90), 3, 1.2 ,0 ,cv2.CV_64F)
gabor_135 = cv2.getGaborKernel((11, 11), 1.5, np.radians(135), 3, 1.2 ,0 ,cv2.CV_64F)

img = cv2.imread("./image_71_80/imori.jpg").astype(np.float)

H, W, C = img.shape
img_gray = BGR2GRAY(img)
img_gray = np.pad(img_gray, (11//2, 11//2), 'edge')

out0 = np.zeros((H, W), dtype=np.float32)
out45 = np.zeros((H, W), dtype=np.float32)
out90 = np.zeros((H, W), dtype=np.float32)
out135 = np.zeros((H, W), dtype=np.float32)

for y in range(H):
    for x in range(W):
        out0[y,x] = np.sum(img_gray[y:y+11, x:x+11] * gabor_0)      #out[0,0]はimg_grayまわりに11//2、0パディングされた画像の[0,0]~[10,10]までの画素（つまり、実際にはimg_gray[-5,-5]~img_gray[5,5]までってこと。img_gray[0,0]が中心）に、ガボールフィルタされたものの合計値が入る。
        out45[y,x] = np.sum(img_gray[y:y+11, x:x+11] * gabor_45)
        out90[y,x] = np.sum(img_gray[y:y+11, x:x+11] * gabor_90)
        out135[y,x] = np.sum(img_gray[y:y+11, x:x+11] * gabor_135)

out0 = np.clip(out0, 0, 255)
out0 = out0.astype(np.uint8)

out45 = np.clip(out45, 0, 255)
out45 = out45.astype(np.uint8)

out90 = np.clip(out90, 0, 255)
out90 = out90.astype(np.uint8)

out135 = np.clip(out135, 0, 255)
out135 = out135.astype(np.uint8)

cv2.imwrite("./image_71_80/answer_79_0.jpg",out0)
cv2.imwrite("./image_71_80/answer_79_45.jpg",out45)
cv2.imwrite("./image_71_80/answer_79_90.jpg",out90)
cv2.imwrite("./image_71_80/answer_79_135.jpg",out135)
cv2.imshow("result", out0)
cv2.waitKey(0)
cv2.destroyAllWindows()
