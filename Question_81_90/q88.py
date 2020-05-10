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


def k_means_step1(db, pdb, Class=2):
    # copy database
    feats = db.copy()

    # initiate random seed
    np.random.seed(1)

    # assign random class ランダムにクラス割当て
    for i in range(len(feats)):
        if np.random.random() < 0.5:
            feats[i, -1] = 0
        else:
            feats[i, -1] = 1

    # prepare gravity 重心割当の配列
    gs = np.zeros((Class, 12), dtype=np.float32)

    # get gravity 各クラスの、各特徴量の重心をだす
    for i in range(Class):
        #feats[np.where(feats[..., -1] == 0)[0], :12] で[0]がないと3次元行列になってしまう。
        gs[i] = np.mean(feats[np.where(feats[..., -1] == i)[0], :12], axis=0)

    print("assigned label")
    print(feats)
    print("Grabity")
    print(gs)

if __name__ == "__main__":
    # get database
    database , train = get_DB()
    k_means_step1(database, train)
