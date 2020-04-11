import cv2
import numpy as np


# average pooling
def average_pooling(img, G=8):
    out = img.copy()

    H, W, C = img.shape
    Nh = int(H / G)
    Nw = int(W / G)

    for y in range(Nh):
        for x in range(Nw):
            for c in range(C):
                out[G*y:G*(y+1), G*x:G*(x+1), c] = np.mean(out[G*y:G*(y+1), G*x:G*(x+1), c]).astype(np.int)
            print(out[y*G:(y+1)*G,x*G:(x+1)*G], y*G , x*G)
    return out


# Read image
img = cv2.imread("./image_01_10/imori.jpg")

# Average Pooling
out = average_pooling(img)

# Save result
cv2.imwrite("./image_01_10/out.jpg", out)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
