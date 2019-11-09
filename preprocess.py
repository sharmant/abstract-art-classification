import cv2
import numpy as np
import pickle
import random

def process_data(pro, amateur):
    data = []
    for i in range(1,6):
        for j in range(1,101):
            num = str(j)
            if len(num) == 1:
                num = "00" + num
            elif len(num) == 2:
                num = "0" + num
            filename = pro+"ppt{0}_{1}.jpg".format(i,num)
            im = cv2.imread(filename,cv2.IMREAD_COLOR)
            res = cv2.resize(im, dsize=(512,512), interpolation=cv2.INTER_CUBIC)
            # print(type(res))
            # print(type(res.mean()))
            res = res - float(res.mean())
            norm_image = cv2.normalize(res, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            data.append(norm_image)

    count = 0
    for i in range(1,441):
        filename = amateur+"da_{}.jpg".format(i)
        im = cv2.imread(filename,cv2.IMREAD_COLOR)
        # print(im)
        # print(filename)
        if im is None:
            count += 1
            # print(filename)
            continue
        res = cv2.resize(im, dsize=(512,512), interpolation=cv2.INTER_CUBIC)
        res = res - float(res.mean())
        norm_image = cv2.normalize(res, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        data.append(norm_image)

    data = np.array(data)
    y = np.concatenate((np.ones(500), np.zeros(440)),  axis=0)
    print("=> Data shape: {}".format(data.shape))
    print("=> y shape: {}".format(y.shape))
    print("=> count: {}".format(count))
    return data, y

def split_data(X, y):
    num_train = int(0.6*X.shape[0])
    num_dev = int(0.2*X.shape[0])
    num_test = X.shape[0] - num_train - num_dev
    print("{} {} {}".format(num_train, num_dev, num_test))
    train_idxs = np.array(random.sample(list(range(X.shape[0])), k=num_train))
    # print(sorted(train_idxs))
    dev_idxs = np.array(random.sample(list(set(range(X.shape[0])) - set(train_idxs)), k=num_dev))
    # print(sorted(dev_idxs))
    test_idxs = np.array(list(set(range(X.shape[0])) - set(train_idxs).union(set(dev_idxs))))
    # print(sorted(test_idxs))
    X_train, y_train = X[train_idxs], y[train_idxs]
    X_dev, y_dev = X[dev_idxs], y[dev_idxs]
    X_test, y_test = X[test_idxs], y[test_idxs]
    print("train: {}, {}".format(X_train.shape, y_train.shape))
    print("dev: {}, {}".format(X_dev.shape, y_dev.shape))
    print("test: {}, {}".format(X_test.shape, y_test.shape))

    print("=> Pickling")
    pickle.dump((X_train, y_train), open("train.p", "wb"))
    pickle.dump((X_dev, y_dev), open("dev.p", "wb"))
    pickle.dump((X_test, y_test), open("test.p", "wb"))
    
    return X_train, y_train, X_dev, y_dev, X_test, y_test

if __name__ == "__main__":
    #if pickled:
        # return pickle.load(open("data.p", "rb"))
    data, y = process_data("abstract_data/mart/", "abstract_data/deviantart/")
    X_train, y_train, X_dev, y_dev, X_test, y_test = split_data(data, y)
    