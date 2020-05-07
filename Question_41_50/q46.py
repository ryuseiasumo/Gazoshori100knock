import cv2
import numpy as np


def Canny(img):
    # Gray scale
    def BGR2GRAY(img):
        b = img[:, :, 0].copy()
        g = img[:, :, 1].copy()
        r = img[:, :, 2].copy()

        # Gray scale
        out = 0.2126 * r + 0.7152 * g + 0.0722 * b
        out = out.astype(np.uint8)

        return out


    # Gaussian filter for grayscale
    def gaussian_filter(img, K_size=3, sigma=1.3):

        if len(img.shape) == 3:
            H, W, C = img.shape
            gray = False
        else:
            img = np.expand_dims(img, axis=-1)
            H, W, C = img.shape
            gray = True

        ## Zero padding
        pad = K_size // 2
        out = np.zeros([H + pad * 2, W + pad * 2, C], dtype=np.float)
        out[pad : pad + H, pad : pad + W] = img.copy().astype(np.float)

        ## prepare Kernel
        K = np.zeros((K_size, K_size), dtype=np.float)
        for x in range(-pad, -pad + K_size):
            for y in range(-pad, -pad + K_size):
                K[y + pad, x + pad] = np.exp( - (x ** 2 + y ** 2) / (2 * sigma * sigma))
        #K /= (sigma * np.sqrt(2 * np.pi))
        K /= (2 * np.pi * sigma * sigma)
        K /= K.sum()

        tmp = out.copy()

        # filtering
        for y in range(H):
            for x in range(W):
                for c in range(C):
                    out[pad + y, pad + x, c] = np.sum(K * tmp[y : y + K_size, x : x + K_size, c])

        out = np.clip(out, 0, 255)
        out = out[pad : pad + H, pad : pad + W]
        out = out.astype(np.uint8)

        if gray:
            out = out[..., 0]

        return out


    # sobel filter
    def sobel_filter(img, K_size=3):
        if len(img.shape) == 3:
            H, W, C = img.shape
        else:
            H, W = img.shape

        # Zero padding
        pad = K_size // 2
        out = np.zeros((H + pad * 2, W + pad * 2), dtype=np.float)
        out[pad : pad + H, pad : pad + W] = img.copy().astype(np.float)
        tmp = out.copy()

        out_v = out.copy()
        out_h = out.copy()

        ## Sobel vertical
        Kv = [[1., 2., 1.],[0., 0., 0.], [-1., -2., -1.]]
        ## Sobel horizontal
        Kh = [[1., 0., -1.],[2., 0., -2.],[1., 0., -1.]]

        # filtering
        for y in range(H):
            for x in range(W):
                out_v[pad + y, pad + x] = np.sum(Kv * (tmp[y : y + K_size, x : x + K_size]))
                out_h[pad + y, pad + x] = np.sum(Kh * (tmp[y : y + K_size, x : x + K_size]))

        out_v = np.clip(out_v, 0, 255)
        out_h = np.clip(out_h, 0, 255)

        out_v = out_v[pad : pad + H, pad : pad + W]
        out_v = out_v.astype(np.uint8)
        out_h = out_h[pad : pad + H, pad : pad + W]
        out_h = out_h.astype(np.uint8)

        return out_v, out_h


    def get_edge_angle(fx, fy):
        # get edge strength
        edge = np.sqrt(np.power(fx.astype(np.float32), 2) + np.power(fy.astype(np.float32), 2))
        edge = np.clip(edge, 0, 255)

        fx = np.maximum(fx, 1e-10)
        #fx[np.abs(fx) <= 1e-5] = 1e-5

        # get edge angle
        angle = np.arctan(fy / fx)

        return edge, angle


    def angle_quantization(angle):
        angle = angle / np.pi * 180
        angle[angle < -22.5] = 180 + angle[angle < -22.5]
        _angle = np.zeros_like(angle, dtype=np.uint8)
        _angle[np.where(angle <= 22.5)] = 0
        _angle[np.where((angle > 22.5) & (angle <= 67.5))] = 45
        _angle[np.where((angle > 67.5) & (angle <= 112.5))] = 90
        _angle[np.where((angle > 112.5) & (angle <= 157.5))] = 135

        return _angle


    def non_maximum_suppression(angle, edge):
        H, W = angle.shape
        _edge = edge.copy()

        for y in range(H):
            for x in range(W):
                    if angle[y, x] == 0:
                            dx1, dy1, dx2, dy2 = -1, 0, 1, 0
                    elif angle[y, x] == 45:
                            dx1, dy1, dx2, dy2 = -1, 1, 1, -1
                    elif angle[y, x] == 90:
                            dx1, dy1, dx2, dy2 = 0, -1, 0, 1
                    elif angle[y, x] == 135:
                            dx1, dy1, dx2, dy2 = -1, -1, 1, 1
                    if x == 0:
                            dx1 = max(dx1, 0)
                            dx2 = max(dx2, 0)
                    if x == W-1:
                            dx1 = min(dx1, 0)
                            dx2 = min(dx2, 0)
                    if y == 0:
                            dy1 = max(dy1, 0)
                            dy2 = max(dy2, 0)
                    if y == H-1:
                            dy1 = min(dy1, 0)
                            dy2 = min(dy2, 0)
                    if max(max(edge[y, x], edge[y + dy1, x + dx1]), edge[y + dy2, x + dx2]) != edge[y, x]:
                            _edge[y, x] = 0

        return _edge

    def hysterisis(edge, HT=100, LT=30):
        H, W = edge.shape

        # Histeresis threshold
        edge[edge >= HT] = 255
        edge[edge <= LT] = 0

        _edge = np.zeros((H + 2, W + 2), dtype=np.float32)
        _edge[1 : H + 1, 1 : W + 1] = edge

        ## 8 - Nearest neighbor
        nn = np.array(((1., 1., 1.), (1., 0., 1.), (1., 1., 1.)), dtype=np.float32)

        for y in range(1, H+2):
                for x in range(1, W+2):
                        if _edge[y, x] < LT or _edge[y, x] > HT:
                                continue
                        if np.max(_edge[y-1:y+2, x-1:x+2] * nn) >= HT:
                                _edge[y, x] = 255
                        else:
                                _edge[y, x] = 0

        edge = _edge[1:H+1, 1:W+1]

        return edge

    # grayscale
    gray = BGR2GRAY(img)

    # gaussian filtering
    gaussian = gaussian_filter(gray, K_size=5, sigma=1.4)

    # sobel filtering
    fy, fx = sobel_filter(gaussian, K_size=3)

    # get edge strength, angle
    edge, angle = get_edge_angle(fx, fy)

    # angle quantization
    angle = angle_quantization(angle)

    # non maximum suppression
    edge = non_maximum_suppression(angle, edge)

    # hysterisis threshold
    out = hysterisis(edge, 100, 30)

    return out

def hough_step3(edge):
    def voting(edge):
        H, W = edge.shape
        drho = 1
        dtheta = 1

        # get rho max length
        rho_max = np.ceil(np.sqrt(H ** 2 + W ** 2)).astype(np.int)

        # hough table
        hough = np.zeros((rho_max * 2, 180), dtype=np.int)

        # get index of edge
        ind = np.where(edge == 255)  #線があるところは白色だから。

        ## hough transformation
        for y, x in zip(ind[0], ind[1]):
            # print(y,x)
            for theta in range(0, 180, dtheta):
                # get polar coordinat4s
                t = np.pi / 180 * theta  #t = π/180 * θ
                rho = int(x * np.cos(t) + y * np.sin(t))
                #ρ = xcos(t) + ysin(t) だ~~~~！！！

                # vote
                hough[rho + rho_max, theta] += 1
                #x座標が角度θ、y座標がrhoの値を表す。rhoは[-rho_max,rho_max]の範囲のため、rho_maxを加えて下駄を履かせてる
                #票がたくさんはいると画素値が上がる、つまり出力画像において明るくなる

        out = hough.astype(np.uint8)

        return out

    def NMS(_img):
        H, W = _img.shape
        img = _img.copy()

        for y in range(H):
            for x in range(W):
                dx1 , dx2 , dy1, dy2 = -1, 2, -1, 2
                if x == 0:
                    dx1 = 0
                if x == W-1:
                    dx2 = 1

                if y == 0:
                    dy1 = 0
                if y == H-1:
                    dy2 = 1

                if img[y][x] >= np.max(img[y+dy1:y+dy2, x+dx1: x+dx2]):
                    continue
                else:
                    img[y][x] = 0

        ind_top20 = np.argsort(img.ravel())[::-1][:20]

        rhos = ind_top20 // 180
        thetas = ind_top20 % 180
        out = np.zeros_like(img, dtype=np.int)
        out[rhos, thetas] = 255

        return out

    def hough_reverse(_img, _hough):
        H, W , C = _img.shape
        img = _img.copy()
        rho_max = np.ceil(np.sqrt(H ** 2 + W ** 2)).astype(np.int)

        ind = np.where(_hough > 0)

        for r, theta in zip(ind[0], ind[1]):
            r -= rho_max
            print(r,theta)
            t = np.pi / 180. * theta
            if np.sin(t) == 0  or np.cos(t) == 0:
                continue

            else:
                for x in range(W):
                    y = int(-np.cos(t)/np.sin(t) * x + r/np.sin(t))
                    # print(y,x)
                    if y > H-1 or y < 0:
                        continue
                    else:
                        print(y,x)
                        img[y,x] = [0,0,255]

                for y in range(H):
                    x = int(-np.sin(t)/np.cos(t) * y + r/np.cos(t))
                    # print(y,x)
                    if x > W-1  or x < 0:
                        continue
                    else:
                        img[y,x] = [0,0,255]

        print("ーーーーーーーーーーー")
        print(img)
        return img


    # voting
    out_1 = voting(edge)
    out_2 = NMS(out_1)
    out = hough_reverse(img, out_2)

    return out



img = cv2.imread("./image_41_50/thorino.jpg").astype(np.float)

edge = Canny(img)
img_ans = hough_step3(edge)

img_ans = img_ans.astype(np.uint8)
cv2.imwrite("./image_41_50/answer46.jpg", img_ans)
cv2.imshow("result", img_ans)
cv2.waitKey(0)
cv2.destroyAllWindows()
