import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

def quantization(img):
    for c in range(3):
        for i in range(4):
            img[..., c][(64*i <= img[..., c]) & (img[..., c] < 64*(i+1))] = i #32 + 64 * i　→ iとすることで、0,1,2,3に割り当てれる

    return img

def get_DB():
    # get image paths
    train = glob("dataset/train_*")
    train.sort()

    db = np.zeros((len(train), 13), dtype=np.int32)

    for i , path in enumerate(train): #enumerate関数を使うと、データと共に、インデックスも取得可能
        img = quantization(cv2.imread(path))
        for j in range(4):
            #青
            db[i,j] = len(img[img[..., 0] == j])

            #緑
            db[i,j+4] = len(img[img[..., 1] == j])

            #赤
            db[i,j+8] = len(img[img[..., 2] == j])

        #get class
        if 'akahara' in path:
            cls = 0
        elif 'madara' in path:
            cls = 1

        # store class label
        db[i, -1] = cls


        img_h = img.copy()
        img_h[..., 1] += 4
        img_h[..., 2] += 8
        print(img_h)
        plt.subplot(2, 5, i+1)
        plt.hist(img_h.ravel(), bins=12, rwidth=0.8)
        plt.title(path)

    print(db)
    plt.show()


if __name__ == "__main__":
    # get database
    get_DB()



# def make_hist(img, class_i = 0):
#     H, W, C= img.shape
#     histogram = np.zeros(13)
#     histogram[12] = class_i

#     #青
#     for i in range(4):
#         img_Bi = img[img[...,0] == i]
#         histogram[i] = len(img_Bi)

#     #緑
#     for i in range(4):
#         img_Gi = img[img[...,1] == i]
#         histogram[i+4] = len(img_Gi)

#     #赤
#     for i in range(4):
#         img_Ri = img[img[...,1] == i]
#         histogram[i+8] = len(img_Ri)

#     return histogram


# for i in range(10):
#     if i < 5:
#         database[i] = make_hist(img_quantization[i], 0)
#         print(database[i])
#         plt.hist(database[i].ravel(), bins = 13, rwidth = 0.8 )
#         plt.show()

#     else:
#         database[i] = make_hist(img_quantization[i], 1)
#         plt.hist(database[i].ravel(), bins = 13, rwidth = 0.8 )
#         plt.show()

# cv2.imwrite("./image_81_90/answer_84.jpg",out)
# cv2.imshow("result", out)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
