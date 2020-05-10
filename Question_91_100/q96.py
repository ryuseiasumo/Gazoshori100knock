import numpy as np
import cv2

np.random.seed(0)

def HOG(img):
    # Grayscale
    def BGR2GRAY(img):
          gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
          return gray

    def get_gradXY(gray):
        H,W= gray.shape
        #辺成分でパディングする
        gray = np.pad(gray,(1,1), 'edge')

        #輝度勾配(傾き)をもとめる
        gx = gray[1:H+1,2:W+2]-gray[1:H+1,0:W]
        gy = gray[2:H+2,1:W+1]-gray[0:H,1:W+1]

        gx[gx == 0] = 1e-6

        return gy, gx

    def get_maggrad(gy, gx):
        #勾配強度と角度
        mag = np.sqrt(gx**2+gy**2)
        ang = np.arctan(gy/gx)
        #量子化に備え、0~πの範囲にする
        ang[ang<0] = ang[ang<0]+np.pi

        return mag, ang

    def quantization(ang):
        gradient_quantized = np.zeros_like(ang)

        for i in range(9):
            gradient_quantized[(i/9*np.pi <= ang) & (ang <=(i+1)/9*np.pi)] = i

        return gradient_quantized

    def gradient_histogram(mag, gradient_quantized, N=8):
        H,W = mag.shape
        # get cell num
        cell_N_H = H // N
        cell_N_W = W // N
        histogram = np.zeros((cell_N_H, cell_N_W, 9), dtype=np.float32)

        for y in range(cell_N_H):
            for x in range(cell_N_W):
                for ang in range(9):
                    histogram[y,x, ang] = np.sum(mag[y*N:y*N+N, x*N: x*N+N][gradient_quantized[y*N:y*N+N, x*N: x*N+N] == ang])

        return histogram


    def normalization(histogram, C = 3, epsilon = 1):
        cell_N_H, cell_N_W , _= histogram.shape
        c = C // 2
        for y in range(cell_N_H):
            for x in range(cell_N_W):
                histogram[y,x] /= np.sqrt(np.sum(histogram[max(y - c, 0) : min(y + c + 1, cell_N_H), max(x - c, 0) : min(x + c + 1 , cell_N_W)] ** 2) + epsilon)

        return histogram

    # 1. BGR -> Gray
    gray = BGR2GRAY(img)

    # 1. Gray -> Gradient x and y
    gx, gy = get_gradXY(gray)

    # 2. get gradient magnitude and angle
    magnitude, gradient = get_maggrad(gx, gy)

    # 3. Quantization
    gradient_quantized = quantization(gradient)

    # 4. Gradient histogram
    histogram = gradient_histogram(gradient_quantized, magnitude)

    # 5. Histogram normalization
    histogram = normalization(histogram)

    return histogram


def IoU(a,b):
    top_l_x = max(a[0], b[0])
    top_l_y = max(a[1], b[1])
    low_r_x = min(a[2],b[2])
    low_r_y = min(a[3], b[3])


    RoL = np.array((top_l_x,top_l_y,low_r_x,low_r_y))
    print(RoL)

    iou_h = RoL[3]-RoL[1]
    iou_w = RoL[2]-RoL[0]

    if iou_h <= 0 or iou_w <= 0:
        return 0.0

    s_RoL = iou_h * iou_w


    s_a = (a[3]-a[1]) * (a[2] - a[0])
    s_b = (b[3]-b[1]) * (b[2] - b[0])


    iou = abs(s_RoL) / abs(s_a + s_b - s_RoL)

    return iou




def resize(img, h, w):
    _h , _w, _c = img.shape

    ah = 1. * h / _h
    aw = 1. * w / _w

    y = np.arange(h).repeat(w).reshape((w,-1))
    x = np.tile(np.arange(w), (h, 1))

    y = y/ah
    x = x/aw

    iy = np.floor(y).astype(np.int32)
    ix = np.floor(x).astype(np.int32)

    # clip index
    ix = np.minimum(ix, _w-2)
    iy = np.minimum(iy, _h-2)


    dy = y - iy
    dx = x - ix

    print(dy)
    print(dx)

    dx = np.tile(dx, [_c, 1, 1]).transpose(1, 2, 0)
    dy = np.tile(dy, [_c, 1, 1]).transpose(1, 2, 0)

    print(dy)
    print(dx)

    out = (1-dx)*(1-dy)*img[iy, ix] + dx*(1-dy)*img[iy,ix+1] +  (1-dx)*dy*img[iy+1, ix] + dx*dy*img[iy+1,ix+1]
    out[out > 255] = 255

    return out




class NN:
    def __init__(self, ind=2, w=64, w2 = 64, outd=1, lr=0.1):
        self.w1 = np.random.normal(0, 1, [ind, w])
        self.b1 = np.random.normal(0, 1, [w])
        self.w2 = np.random.normal(0, 1, [w, w2])
        self.b2 = np.random.normal(0, 1, [w2])
        self.wout = np.random.normal(0, 1, [w2, outd])
        self.bout = np.random.normal(0, 1, [outd])
        self.lr = lr

    def forward(self, x):
        self.z1 = x
        self.z2 = sigmoid(np.dot(self.z1, self.w1) + self.b1)
        self.z3 = sigmoid(np.dot(self.z2, self.w2) + self.b2)
        self.out = sigmoid(np.dot(self.z3, self.wout) + self.bout)
        return self.out

    def train(self, x, t):
        # backpropagation output layer
        #En = t * np.log(self.out) + (1-t) * np.log(1-self.out)
        En = (self.out - t) * self.out * (1 - self.out)
        grad_En = En #np.array([En for _ in range(t.shape[0])])
        grad_wout = np.dot(self.z3.T, En)
        grad_bout = np.dot(np.ones([En.shape[0]]), En)
        self.wout -= self.lr * grad_wout#np.expand_dims(grad_wout, axis=-1)
        self.bout -= self.lr * grad_bout

        # backpropagation inter layer2
        grad_u2 = np.dot(En, self.wout.T) * self.z3 * (1 - self.z3)
        grad_w2 = np.dot(self.z2.T, grad_u2)
        grad_b2 = np.dot(np.ones([grad_u2.shape[0]]), grad_u2)
        self.w2 -= self.lr * grad_w2
        self.b2 -= self.lr * grad_b2

        # backpropagation inter layer1
        grad_u1 = np.dot(grad_u2, self.w2.T) * self.z2 * (1 - self.z2)
        grad_w1 = np.dot(self.z1.T, grad_u1)
        grad_b1 = np.dot(np.ones([grad_u1.shape[0]]), grad_u1)
        self.w1 -= self.lr * grad_w1
        self.b1 -= self.lr * grad_b1

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

#訓練
def train_nn(nn, train_x, train_t, interation_N = 10000):
    for i in range(interation_N):
        nn.forward(train_x)
        nn.train(train_x, train_t)
    return nn

#テスト
def test_nn(nn, test_x, test_t, pred_th = 0.5):
    accuracy_N = 0.
    for data, t in zip(test_x, test_t):
        prob = nn.forward(data)

        if prob >= pred_th:
            pred = 1
        else:
            pred = 0

        if t == pred:
            accuracy_N += 1

    accuracy = accuracy_N / len(db)

    print("Accuracy >> {} ({} / {})".format(accuracy, accuracy_N, len(db)))




def make_dataset(img, gt, K = 200,L= 60, th = 0.5, H_size = 32):
    H, W, C = img.shape

    # get HOG feature dimension
    HOG_feature_N = ((H_size // 8) ** 2) * 9

    # prepare database
    db = np.zeros([K, HOG_feature_N + 1])


    for i in range(K):
        x1 = np.random.randint(W-L)
        y1 = np.random.randint(H-L)
        x2 = x1 + L
        y2 = y1 + L

        crop = np.array((x1,y1,x2,y2),dtype = np.float32)

        iou = IoU(gt,crop)

        if iou >= th:
            cv2.rectangle(img, (x1,y1),(x2, y2),(0,0,255), 1)
            label = 1

        elif iou < th:
            cv2.rectangle(img, (x1,y1),(x2, y2),(255,0,0), 1)
            label = 0

        # 領域
        crop_area = img[y1:y2, x1:x2]

        # resize crop area
        crop_area = resize(crop_area, H_size, H_size)

        # get HOG feature
        _hog = HOG(crop_area)

        # store HOG feature and label
        db[i, :HOG_feature_N] = _hog.ravel()
        db[i, -1] = label

    return db




img = cv2.imread("./image_91_100/imori.jpg").astype(np.float32)

histogram = HOG(img)
# prepare gt bounding box
gt = np.array((47, 41, 129, 103), dtype=np.float32)

# get database
db = make_dataset(img, gt)

# train neural network
# get input feature dimension
input_dim = db.shape[1] - 1
# prepare train data X
train_x = db[:, :input_dim]
# prepare train data t
train_t = db[:, -1][..., None]


nn = NN(ind=input_dim, lr = 0.01)

nn = train_nn(nn, train_x, train_t, interation_N = 10000)

test_nn( nn, train_x, train_t)
