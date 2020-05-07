import cv2
import numpy as np

def Gaborfilter(K_size=111, Sigma=10, Gamma=1.2, Lambda=10, Psi=0, angle=0):
    # get half size
    d = K_size // 2

    # prepare kernel
    gabor = np.zeros((K_size, K_size), dtype=np.float32)


    # each value
    for y in range(K_size):
        for x in range(K_size):
            # distance from center
            px = x - d
            py = y - d

            # degree -> radian
            theta = angle / 180. * np.pi

            # get kernel x
            _x = np.cos(theta) * px + np.sin(theta) * py

            # get kernel y
            _y = -np.sin(theta) * px + np.cos(theta) * py

            # fill kernel
            gabor[y, x] = np.exp(-(_x**2 + Gamma**2 * _y**2) / (2 * Sigma**2)) * np.cos(2*np.pi*_x/Lambda + Psi)


    gabor /= np.sum(np.abs(gabor))

    return gabor


out = Gaborfilter(K_size=111, Sigma=10, Gamma=1.2, Lambda=10, Psi=0, angle=90)


out = out - np.min(out)
out /= np.max(out)
out *= 255

img_ans = out.astype(np.uint8)

cv2.imwrite("./image_71_80/answer_78.jpg",img_ans)
cv2.imshow("result", img_ans)
cv2.waitKey(0)
cv2.destroyAllWindows()
