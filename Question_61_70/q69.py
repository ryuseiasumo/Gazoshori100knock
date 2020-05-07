import cv2
import numpy as np
import matplotlib.pyplot as plt

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

    #ヒストグラムは、各セルごとに、各角度分用意する。
    histogram = np.zeros((cell_N_H, cell_N_W, 9)).astype(np.float32)

    for y in range(cell_N_H):
        for x in range(cell_N_W):
            #それぞれのセルごとにみる

            #愚直for文型
            # for ty in range(N):
            #     for tx in range(N):
            #         ang = arg_quantized[y*N+ty, x*N+tx]
            #         histogram[y,x,ang] += mag[y*N+ty, x*N+tx]

            for ang in range(9):
                histogram[y,x, ang] = np.sum(mag[y*N:y*N+N, x*N: x*N+N][arg_quantized[y*N:y*N+N, x*N: x*N+N] == ang])

    return histogram

def HOG_step3(hist, epsilon = 1):
    cell_N_H, cell_N_W, _ = hist.shape
    ## each histogram
    for y in range(cell_N_H):
        for x in range(cell_N_W):
            hist[y, x] /= np.sqrt(np.sum(hist[max(y - 1, 0) : min(y + 2, cell_N_H),max(x - 1, 0) : min(x + 2, cell_N_W)] ** 2) + epsilon)

    print(hist)
    return hist

def HOG_step4(img, histogram):
    # Grayscale
    def BGR2GRAY(img):
        gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
        return gray

    def draw(gray, histogram, N=8):
        # get shape
        H, W = gray.shape
        cell_N_H, cell_N_W, _ = histogram.shape

        ## Draw
        out = gray[1 : H + 1, 1 : W + 1].copy().astype(np.uint8)

        for y in range(cell_N_H):
            for x in range(cell_N_W):
                cx = x * N + N // 2
                cy = y * N + N // 2
                x1 = cx + N // 2 - 1
                y1 = cy
                x2 = cx - N // 2 + 1
                y2 = cy

                h = histogram[y, x] / np.sum(histogram[y, x])
                h /= h.max()

                for c in range(9):
                    #angle = (20 * c + 10 - 90) / 180. * np.pi
                    # get angle
                    angle = (20 * c + 10) / 180. * np.pi
                    rx = int(np.sin(angle) * (x1 - cx) + np.cos(angle) * (y1 - cy) + cx)
                    ry = int(np.cos(angle) * (x1 - cx) - np.cos(angle) * (y1 - cy) + cy)
                    lx = int(np.sin(angle) * (x2 - cx) + np.cos(angle) * (y2 - cy) + cx)
                    ly = int(np.cos(angle) * (x2 - cx) - np.cos(angle) * (y2 - cy) + cy)

                    # color is HOG value
                    c = int(255. * h[c])

                    # draw line
                    cv2.line(out, (lx, ly), (rx, ry), (c, c, c), thickness=1)

        return out

    # get gray
    gray = BGR2GRAY(img)

    # draw HOG
    out = draw(gray, histogram)

    return out

img = cv2.imread("./image_61_70/imori.jpg").astype(np.float32)
img_gray = BGR2GRAY(img)

#1 ~ 3
magnitude, gradient_quantized = HOG_step1(img_gray)

#4
histogram_no_Normalizetion = HOG_step2(img_gray,magnitude, gradient_quantized, 8)

#5
histogram = HOG_step3(histogram_no_Normalizetion)

#矢印を書く
img_HOG = HOG_step4(img, histogram)

cv2.imwrite("./image_61_70/answer_69.jpg", img_HOG)
cv2.imshow("result" , img_HOG)
cv2.waitKey(0)
cv2.destroyAllWindows()
