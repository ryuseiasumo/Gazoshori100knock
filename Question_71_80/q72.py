import cv2
import numpy as np

def get_musk(img):
    #BGR -> HSV
    # print(img)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    musk = np.zeros_like(img_hsv)
    musk[(img_hsv[:,:,0] >= 90) & (img_hsv[:,:, 0] <= 130)] = 1
    return musk

def musking(_musk ,img):
    musk = 1 - _musk
    out = musk * img
    out = out.astype(np.uint8)
    return out


def Morph_open(_img , time = 1):
    # kernel
    kernel = np.array(((0, 1, 0),
                (1, 0, 1),
                (0, 1, 0)), dtype=np.uint8)

    erosion = cv2.erode(_img,kernel,iterations = time)
    dilation = cv2.dilate(erosion,kernel,iterations = time)
    return dilation


def Morph_close(_img , time = 1):
    kernel = np.array(((0, 1, 0),
                (1, 0, 1),
                (0, 1, 0)), dtype=np.uint8)

    dilation = cv2.dilate(_img,kernel,iterations = time)
    erosion = cv2.erode(dilation,kernel,iterations = time)
    return erosion


img = cv2.imread("./image_71_80/imori.jpg")
musk = get_musk(img)

musk_c = Morph_close(musk, 5)
musk_o = Morph_open(musk_c, 5)

img_ans = musking(musk_o, img)


cv2.imwrite("./image_71_80/answer_72.jpg",img_ans)
cv2.imshow("result", img_ans)
cv2.waitKey(0)
cv2.destroyAllWindows()
