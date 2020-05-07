import cv2
import numpy as np

def Gray(img_color):
    R = img_color[:,:,2].copy()
    G = img_color[:,:,1].copy()
    B = img_color[:,:,0].copy()

    img_gray = 0.2126 * R + 0.7152 * G + 0.0722 * B
    img_gray = img_gray.astype(np.uint8)
    return img_gray

if __name__ == "__main__":
    img = cv2.imread("./image_01_10/imori.jpg")
    img_ans = Gray(img)

    cv2.imwrite("./image_01_10/answers2.jpg", img_ans)
    cv2.imshow("result", img_ans)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
