import cv2
import numpy as np

def step1(img, K = 1):
    H, W ,C= img.shape
    np.random.seed(0)

    # ーーーーーー  サンプリング(手順1)  ーーーーーーー

    #二次元配列に変換
    img = np.reshape(img, (H*W, -1))
    # K個抽出
    i = np.random.choice(np.arange(H*W), K, replace=False)
    img_choice = img[i].copy()
    print(img_choice)

    # np.random.choiceが使える配列は一次元配列のみ！→ imgは、色の表現があるため、どうしても二次元配列までしかできない。だからH*Wからインデックスを取り出して、imgからその選ばれたインデックスの部分を抽出する。なので、下のようにはできない。
    # img = np.reshape(img, (H*W, -1))
    # img_choice = np.random.choice(img, K, replace=False)

    # ーーーーーー  (手順 2)  ーーーーーーー
    #二次元配列を作る。
    clss = np.zeros((H * W), dtype=int)

    for i in range(H*W):
        #各画像の各色の、サンプリングした画素たちからの距離の合計を求める。
        # 色の距離 dis = sqrt( (R-R')^2 + (G-G')^2 + (B-B')^2)において、R,G,BにはサンプリングされたK個の画素値が、R',G',B'にはimg[i]の画素値がそれぞれはいる
        dis = np.sqrt(np.sum((img_choice - img[i])**2, axis = 1))

        #最も距離の近いサンプリング画素はどれかをclssに格納する
        #画素1 = 0,画素2 = 1,画素3 = 2,画素4 = 3, 画素5 = 4と割り振る。
        clss[i] = np.argmin(dis)
    print(clss)


    # show
    # 50をかけることで違いがわかりやすくなる
    # 値が0のところは、画素1の色に最もちかいってこと！
    out = np.reshape(clss, (H, W)) * 50
    out = out.astype(np.uint8)

    return out



img = cv2.imread("./image_91_100/imori.jpg").astype(np.float)
img_ans = step1(img,K = 5)


cv2.imwrite("./image_91_100/answer_91.jpg", img_ans)
cv2.imshow("result", img_ans)
cv2.waitKey(0)
cv2.destroyAllWindows()
