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
        # print(img_h)
        plt.subplot(2, 5, i+1)
        plt.hist(img_h.ravel(), bins=12, rwidth=0.8)
        plt.title(path)

    # print(db)
    plt.show()
    return db, train

def KNN(db, train):
    test = glob("dataset/test_*")
    test.sort()

    d = np.zeros(13)

    setdata = []

    for i, path in enumerate(test):
        img = quantization(cv2.imread(path))
        for j in range(4):
            d[j] = len(img[img[..., 0] == j])
            d[j + 4] = len(img[img[..., 1] == j])
            d[j + 8] = len(img[img[..., 2 ] == j])

        similar_meter = np.zeros((len(db),12))
        for k in range(len(db)):
            similar_meter[k] = np.abs(d[0:12] - db[k][0:12])

        # ind = np.argmin(np.sum(similar_meter,axis=1))
        ind_sort = np.argsort(np.sum(similar_meter,axis=1))
        # print(similar_meter)
        # print(ind)

        c0 = 0
        c1 = 0
        for k in range(3):
            if db[ind_sort[k]][12] == 0:
                c0 += 1
            elif db[ind_sort[k]][12] == 1:
                c1 += 1

        Cls = "測定不能"
        if c0 > c1:
            Cls = "akahara"
        elif c0 < c1:
            Cls = "madara"

        print(path , "is similar >>", train[ind_sort[0]] , train[ind_sort[1]] ,  train[ind_sort[2]] ,  "Pred >>" , Cls)

        setdata.append([path, Cls])

    return setdata

def Accuracy(data):
    acc = 0
    acc_N = len(data)
    for i in range(len(data)):
        path = data[i][0]
        cls = data[i][1]
        if cls in path:
            acc += 1

    print("Accuracy >>", acc/acc_N)


if __name__ == "__main__":
    # get database
    database , train = get_DB()
    setdata = KNN(database, train)
    print(setdata)
    Accuracy(setdata)
