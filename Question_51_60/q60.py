import cv2
import numpy as np

img1 = cv2.imread("./image_51_60/imori.jpg").astype(np.float)
img2 = cv2.imread("./image_51_60/thorino.jpg").astype(np.float)


alpha = 0.6
out = img1 * alpha + img2 * (1 - alpha)
out = out.astype(np.uint8)

cv2.imwrite("./image_51_60/answer_60.jpg", out)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
