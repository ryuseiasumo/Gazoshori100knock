import cv2
img = cv2.imread("./image_01_10/imori.jpg")
ans_img = img[:, :,(2,1,0)].copy()
print(ans_img.shape)

cv2.imwrite("./image_01_10/answer1.jpg",ans_img)


#範囲を指定してないから、一次元減ってしまう。またチャネル数が1になるため、グレースケールだとみなされる
red_jigensakujen = img[:, :, 2].copy()
cv2.imwrite("./image_01_10/img_red_jigensakujen.jpg",red_jigensakujen)
print(red_jigensakujen.shape)

#次元は減らないがやってることは一緒。またチャネル数が1になるため、グレースケールだとみなされる
red_notjigensakujen = img[:, :, 2:3].copy()
cv2.imwrite("./image_01_10/img_red_notjigensakujen.jpg",red_notjigensakujen)
print(red_notjigensakujen.shape)

#チャネル数を減らさず、BGの要素を0にすることで、赤だけ取り出す
red = img.copy()
red[:,:,(0,1)] = 0
cv2.imwrite("./image_01_10/img_red.jpg",red)
print(red.shape)


# import cv2

# # function: BGR -> RGB
# def BGR2RGB(img):
#     b = img[:, :, 0].copy()
#     g = img[:, :, 1].copy()
#     r = img[:, :, 2].copy()

#     # RGB > BGR
#     img[:, :, 0] = r
#     img[:, :, 1] = g
#     img[:, :, 2] = b

#     return img

# # Read image
# img = cv2.imread("imori.jpg")

# # BGR -> RGB
# img = BGR2RGB(img)

# # Save result
# cv2.imwrite("out.jpg", img)
# cv2.imshow("result", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
