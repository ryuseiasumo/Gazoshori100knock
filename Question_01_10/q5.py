import cv2
import numpy as np

def convert_hsv(_img):
    img = _img.copy()/255.
    hsv = np.zeros_like(img, dtype=np.float32)

	# get max and min
    max_v = img.max(axis = 2)
    min_v = img.min(axis = 2)
    min_arg = img.argmin(axis = 2)

	# H
    hsv[:,:,0][np.where(max_v == min_v)] = 0

    # if min == B
    ind_b = np.where(min_arg == 0)
    hsv[:,:,0][ind_b] = 60 * (img[:,:,1][ind_b] - img[:,:,2][ind_b]) / (max_v[ind_b] - min_v[ind_b]) + 60

	## if min == G
    ind_g = np.where(min_arg == 1)
    hsv[:,:,0][ind_g] = 60 * (img[:,:,2][ind_g] - img[:,:,0][ind_g]) / (max_v[ind_g] - min_v[ind_g]) + 300

    ## if min == R
    ind_r = np.where(min_arg == 2)
    hsv[:,:,0][ind_r] = 60 * (img[:,:,0][ind_r] - img[:,:,1][ind_r]) / (max_v[ind_r] - min_v[ind_r]) + 180

    # S
    hsv[:,:,1] = max_v.copy() - min_v.copy()

    hsv[:,:,2] = max_v.copy()

    return hsv

def convert_rgb(_img, img_hsv):
    img = _img.copy() / 255.

    max_v = img.max(axis = 2).copy()
    min_v = img.min(axis = 2).copy()

    img_ans = np.zeros_like(img)

    H = img_hsv[:,:,0]
    S = img_hsv[:,:,1]
    V = img_hsv[:,:,2]

    C = S
    H_ = H / 60.
    X = C * (1 - np.abs( H_ % 2 - 1))
    Z = np.zeros_like(H)

    vals = [[Z,X,C], [Z,C,X], [X,C,Z], [C,X,Z], [C,Z,X], [X,Z,C]]

    for i in range(6):
        ind = np.where((i <= H_) & (H_ < (i+1)))
        img_ans[:,:,0][ind] = (V-C)[ind] + vals[i][0][ind]  #?
        img_ans[:,:,1][ind] = (V-C)[ind] + vals[i][1][ind]  #?
        img_ans[:,:,2][ind] = (V-C)[ind] + vals[i][2][ind]  #?

    img_ans[np.where(max_v == min_v)] = 0  #?

    img_ans = (img_ans * 255).astype(np.uint8)
    return img_ans


# Read image
img = cv2.imread("./image_01_10/imori.jpg").astype(np.float32)

# RGB > HSV
hsv = convert_hsv(img)

# Transpose Hue
hsv[:,:, 0] = (hsv[:,:, 0] + 180) % 360
# print(hsv[:,:,0])

# HSV > RGB
img_ans = convert_rgb(img, hsv)

fp = open("./image_01_10/answer5.txt", "w")
fp.write(str(list(zip(img_ans[2],np.where(img_ans)[0],np.where(img_ans)[1]))))
fp.close()

# Save result
cv2.imwrite("./image_01_10/answer5.jpg", img_ans)
cv2.imshow("result", img_ans)
cv2.waitKey(0)
cv2.destroyAllWindows()
