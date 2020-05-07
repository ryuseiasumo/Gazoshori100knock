import cv2
import numpy as np

img = cv2.imread("./image_61_70/imori.jpg")
H, W, _= img.shape

img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
out = np.zeros((H, W))

out[(90 <= img_hsv[..., 0] )& ( img_hsv[..., 0]<= 130)] = 255

cv2.imwrite("./image_61_70/answer70.jpg", out)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
