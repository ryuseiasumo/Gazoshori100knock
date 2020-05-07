import cv2
import numpy as np

def BGR2GRAY(img):
    # Grayscale
    gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
    return gray



def HOG_step1(gray):
    H, W = gray.shape

    # padding before grad
    gray = np.pad(gray, (1, 1), 'edge')

   # 1. Gray -> Gradient x and y
    # get grad x
    gx = gray[1:H+1, 2:] - gray[1:H+1, :W]
    # get grad y
    gy = gray[2:, 1:W+1] - gray[:H, 1:W+1]
    # replace 0 with
    gx[gx == 0] = 1e-6

    # 2. get gradient magnitude and angle
    mag = np.sqrt(gx**2 + gy**2)
    ang = np.arctan(gy/ gx)

    #0~πの範囲にしたい。tanはπ/2周期のため、tan(θ) = tan(θ+π)より、πを足せばいい。
    ang[ang < 0] = ang[ang < 0] + np.pi


    # 3. Quantization
    #量子化
    ang_quantized = np.zeros_like(ang, dtype=np.int)
    #π/9ごとに分割
    for i in range(9):
        ang_quantized[(np.pi/9 * i <= ang) & (ang < np.pi/9 * (i+1))] = i

    return mag, ang_quantized

def HOG_step2(img_gray, mag, arg_quantized, N = 8):
    H, W = img_gray.shape

    #NxNの、セルの個数
    cell_N_H = H // N
    cell_N_W = W // N

    histogram = np.zeros((cell_N_H, cell_N_W, 9)).astype(np.float32)
    




    for  in expression_list:
        pass


img = cv2.imread("./image_61_70/imori.jpg").astype(np.float32)
img_gray = BGR2GRAY(img)

#1 ~ 3
magnitude, gradient_quantized = HOG_step1(img_gray)

#4
histogram = HOG_step2(img_gray,magnitude, gradient_quantized, 8)


# Write gradient magnitude to file
_magnitude = (magnitude / magnitude.max() * 255).astype(np.uint8)

# Write gradient angle to file
H, W, C = img.shape
out = np.zeros((H, W, 3), dtype=np.uint8)

# define color
C = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255],
    [127, 127, 0], [127, 0, 127], [0, 127, 127]]

# draw color
for i in range(9):
    out[gradient_quantized == i] = C[i]


cv2.imwrite("./image_61_70/answer_67.jpg", _magnitude)
cv2.imwrite("./image_61_70/answer_67_gra.jpg", out)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
