import cv2
import numpy as np
import matplotlib.pyplot as plt


def Gaborfilter(g = 1.2, l = 10, p = 0, A = 0, s = 10, K_size = 111):
    d = K_size//2

    out = np.zeros((K_size, K_size), dtype=np.float32)

    for y in range(K_size):
        for x in range(K_size):
            py = y - d
            px = x - d

            theta = A / 180. * np.pi

            tx = np.cos(theta) * px + np.sin(theta) * py
            ty = -np.sin(theta) * px + np.cos(theta) * py

            out[y, x] = np.exp(-(tx**2 + g**2 * ty**2) /(2*s**2)) * np.cos(2*np.pi*tx/l + p )

    #nomalization
    out /= np.sum(np.abs(out))
    print(out)

    return out

# Read image

out = Gaborfilter()


out = out - np.min(out)
out /= np.max(out)
out *= 255

img_ans = out.astype(np.uint8)

cv2.imwrite("./image_71_80/answer_77.jpg",img_ans)
cv2.imshow("result", img_ans)
cv2.waitKey(0)
cv2.destroyAllWindows()
