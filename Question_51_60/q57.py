import cv2
import numpy as np

def template_matching(_img, _template):
    H, W ,C = _img.shape
    th, tw, tc = _template.shape

    R = np.zeros((H-th, W-tw))

    for y in range(H-th):
        for x in range(W-tw):
            mt = np.mean(_template[0:th,0:tw])
            mi = np.mean(_img[y:y+th, x:x+tw])
            s1 = np.sum((_template[0:th,0:tw]-mt) * (_img[y:y+th, x:x+tw]-mi))
            s2 = np.sqrt(np.sum((_template[0:th, 0:tw] - mt)** 2)) * np.sqrt(np.sum((_img[y:y+th, x:x+tw]-mi )** 2))
            R[y ,x] = s1/s2

    print(R)
    min_loc = np.unravel_index(np.argmax(R), R.shape)
    min_y, min_x = min_loc
    min_loc = (min_x, min_y)
    max_loc = (min_x + tw, min_y+th)

    img_ans = cv2.imread("./image_51_60/imori.jpg")
    cv2.rectangle(img_ans,min_loc, max_loc, (0, 0, 255), thickness= 1)

    return img_ans.astype(np.uint8)

img = cv2.imread("./image_51_60/imori.jpg").astype(np.float)
template = cv2.imread("./image_51_60/imori_part.jpg").astype(np.float)

img_ans = template_matching(img, template)

cv2.imwrite("./image_51_60/answer57.jpg",img_ans)
cv2.imshow("result", img_ans)
cv2.waitKey(0)
cv2.destroyAllWindows()
