import cv2
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("..")
from Question_01_10.q2 import Gray

img = cv2.imread("./image_11_20/imori_dark.jpg")

plt.hist(img.ravel(), bins = 255, rwidth = 0.8, range=(0, 255))
plt.savefig("out.png")
plt.show()

img_g = Gray(img)
plt.hist(img_g.ravel(), bins = 255, rwidth = 0.8, range=(0,255))
plt.savefig("out_g.png")
plt.show()
