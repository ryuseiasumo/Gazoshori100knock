import cv2
import numpy as np

def step2(img, K = 1):
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

    # ーーーーーー  (手順 4)  ーーーーーーー
    #各クラスの画素値が変化しなくなるまで、繰り返す
    while True:
        # cnt = 0 (手順4のやり方その2 (わかりやすいが時間かかる))
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


        # ーーーーーー  (手順 3)  ーーーーーーー
        #各インデックスに対応する色成分の平均をRGBそれぞれに対して取り、新たなクラスとする。
        #つまり、画素1に近いと割り当てられた画素たちの、本来の画素値の平均をとり、それを画素1の新しい画素値(つまりは代表値)とする。
        old = img_choice
        new = np.zeros_like(img_choice)

        for i in range(K):
            #axis = 0にしないと、new[i]がB,G,Rすべての値を平均したものとなる。引数axisに0を渡すと列ごとの平均値、1を渡すと行ごとの平均値が得られる。
            new[i] = np.mean(img[clss == i],axis = 0)

        print(old)
        print(new)

        if (old == new).all():
            break

        else:
            img_choice = new

        #(手順4のやり方その2 (わかりやすいが時間かかる))
        # cnt = 0
        # for i in range(K):
        #     old = img_choice[i]
        #     new = np.mean(img[clss == i],axis = 0)
        #     print(old)
        #     print(new)
        #     diff = np.sum(old - new)
        #     print(diff)
        #     if (old == new).all():
        #         continue
        #     else:
        #         img_choice[i] = new
        #         cnt +=1

        # print(img_choice)
        # if cnt < 1:
        #     break

    # ーーーーーー  (手順 5)  ーーーーーーー
    img_clss = np.reshape(clss, (H, W))
    out = np.zeros((H,W,C))

    for i in range(K):
        out[img_clss == i] = img_choice[i]

    out = out.astype(np.uint8)

    return out



img = cv2.imread("./image_91_100/imori.jpg").astype(np.float)
img_m = cv2.imread("./image_91_100/madara.jpg").astype(np.float)

img_ans = step2(img,K = 5)
img_ans2 = step2(img,K = 10)
img_ans3 = step2(img_m,K = 5)

cv2.imwrite("./image_91_100/answer_92.jpg", img_ans)
cv2.imwrite("./image_91_100/answer_92_k10.jpg", img_ans2)
cv2.imwrite("./image_91_100/answer_92_m.jpg", img_ans3)
cv2.imshow("result", img_ans)
cv2.waitKey(0)
cv2.destroyAllWindows()
