import cv2
import numpy as np
import matplotlib.pyplot as plt


# Grayscale
def BGR2GRAY(img):
    # Grayscale
    gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
    return gray


def Biliner(img, magnification):
    H, W  = img.shape
    new_H = int(H * magnification)
    new_W = int(W * magnification)

    y = np.arange(new_H).repeat(new_W).reshape((new_H,-1))
    x = np.tile(np.arange(new_W), (new_H,1))

    ty = y/magnification
    tx = x/magnification

    iy = np.floor(ty).astype(np.uint8)
    ix = np.floor(tx).astype(np.uint8)

    ix = np.minimum(ix, W-2)
    iy = np.minimum(iy, H-2)

    dy = ty - iy
    dx = tx - ix

    out = (1-dx) * (1-dy) * img[iy, ix] + dx * (1 - dy) * img[iy, ix+1] + (1 - dx) * dy * img[iy+1, ix] + dx * dy * img[iy+1, ix+1]

    out = out.clip(0,255)
    out = out.astype(np.uint8)
    return out

# make image pyramid
def make_pyramid(gray):
    # first element
    pyramid = [gray]
    # each scale
    for i in range(1, 6):
        # define scale
        a = 2. ** i

        # down scale
        p = Biliner(gray, 1./a)


        p_risize = Biliner(p, a)
        pyramid.append(p_risize.astype(np.float32))

    return pyramid

def diff_image(pyramid, i, j):
    out = np.abs(pyramid[i] - pyramid[j])
    return out

# Read image
img = cv2.imread("./image_71_80/imori.jpg").astype(np.float)
gray = BGR2GRAY(img)

# pyramid
pyramid = make_pyramid(gray)

H, W, _ = img.shape
out = np.zeros((H, W), dtype=np.float32)
out += diff_image(pyramid, 0,1)
out += diff_image(pyramid, 0,3)
out += diff_image(pyramid, 0,5)
out += diff_image(pyramid, 1,4)
out += diff_image(pyramid, 2,3)
out += diff_image(pyramid, 3,5)

out = out/out.max()* 255.
img_ans = out.astype(np.uint8)

cv2.imwrite("./image_71_80/answer_76.jpg",img_ans)
cv2.imshow("result", img_ans)
cv2.waitKey(0)
cv2.destroyAllWindows()
