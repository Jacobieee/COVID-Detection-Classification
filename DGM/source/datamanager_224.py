import numpy as np
# import tensorflow as tf
import cv2
from glob import glob
import os
import pandas as pd
from skimage import exposure

from sklearn.utils import shuffle


class Dataset(object):

    def __init__(self, normalize=True):

        print("\nInitializing Dataset...")

        self.normalize = normalize

        # normals, n_labels, abnormals, ab_labels = self.get_other()
        # assert(normals.shape[0] == 20672)
        # assert(n_labels.shape[0] == 20672)
        # assert(abnormals.shape[0] == 6012)
        # assert(ab_labels.shape[0] == 6012)

        # train_data = np.concatenate((normals, abnormals), axis=0)
        # train_label = np.concatenate((n_labels, ab_labels), axis=0)
        # print(train_data.shape)
        # print(train_label.shape)
        # train_data, train_label = shuffle(train_data, train_label)
        # assert(train_data.shape() == (26684, 224, 224))
        # assert(train_label.shape() == (26684))


        # data_train = {
        #   "x_train": train_data,
        #   "y_train": train_label
        # }
        # np.savez("train_rsna.npz", **data_train)






        # read images.
        data = self.load_data()
        # read labels.
        df = pd.read_csv("./features_v4.csv")
        classes = df[['id', 'label']]
        # print(classes)
        imgs = np.zeros(shape=[6334, 224, 224])
        labels = np.zeros(shape=[6334])

        for i, d in enumerate(data):
            img = cv2.imread(d, 0)
            img = cv2.resize(img, (224, 224))
            img = cv2.equalizeHist(img)
            imgs[i] = img
            # print(d)
            cls = classes[classes["id"] == d[-16:-4]]

            if str(cls['label'][:4]).split(" ")[4] == "none":
                
                label = 1
            else:
                label = 0

            labels[i] = label

        # here we split 4500 samples for training, and 1834 for testing.
        self.x_tr, self.y_tr = imgs[:4500], labels[:4500]
        self.x_te, self.y_te = imgs[4500:], labels[4500:]

        self.x_tr = np.ndarray.astype(self.x_tr, np.float32)
        self.x_te = np.ndarray.astype(self.x_te, np.float32)

        print(self.x_tr.shape)
        print(self.x_te.shape)
        # # print(self.y_te[:30])
        self.split_dataset()

        self.num_tr, self.num_te = self.x_tr.shape[0], self.x_te.shape[0]
        self.idx_tr, self.idx_te = 0, 0

        print("Number of data\nTraining: %d, Test: %d\n" %(self.num_tr, self.num_te))

        x_sample, y_sample = self.x_te[0], self.y_te[0]
        self.height = x_sample.shape[0]
        self.width = x_sample.shape[1]
        try: self.channel = x_sample.shape[2]
        except: self.channel = 1

        self.min_val, self.max_val = x_sample.min(), x_sample.max()
        # self.num_class = (y_te.max()+1)
        self.num_class = 2

        print("Information of data")
        print("Shape  Height: %d, Width: %d, Channel: %d" %(self.height, self.width, self.channel))
        print("Value  Min: %.3f, Max: %.3f" %(self.min_val, self.max_val))
        print("Class  %d" %(self.num_class))
        print("Normalization: %r" %(self.normalize))
        if(self.normalize): print("(from %.3f-%.3f to %.3f-%.3f)" %(self.min_val, self.max_val, 0, 1))

        data_train = {
          "x_train": self.x_tr,
          "y_train": self.y_tr
        }
        np.savez("train_hist.npz", **data_train)
        data_test = {
          "x_test": self.x_te,
          "y_test": self.y_te
        }
        np.savez("test_hist.npz", **data_test)


    def get_mask(self, image):
        output = np.zeros(image.shape, np.uint8)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if image[i][j] < 15:
                    output[i][j] = 127
                else:
                    output[i][j] = 0

        output = cv2.blur(output, (3, 3))
        return output

    def split_dataset(self):
        # all images and labels.
        x_total = np.append(self.x_tr, self.x_te, axis=0)
        y_total = np.append(self.y_tr, self.y_te, axis=0)

        x_normal, y_normal = [], []
        x_abnormal, y_abnormal = [], []
        # for all labels.
        for yidx, y in enumerate(y_total):

            y_tmp = np.expand_dims(y_total[yidx], axis=0)

            if(y == 1):     # as normal
                x_normal.append(x_total[yidx])
                y_normal.append(y_tmp)
            else:   # as abnormal
                x_abnormal.append(x_total[yidx])
                y_abnormal.append(y_tmp)

            # if((len(x_normal) >= 2000) and len(x_abnormal) >= 2000): break
        # print(len(x_normal), len(y_normal))
        # print(len(x_abnormal), len(y_abnormal))

        # x_normal, y_normal = x_normal[:2000], y_normal[:2000]
        # x_abnormal, y_abnormal = x_abnormal[:2000], y_abnormal[:2000]

        # We take all data instead.
        x_normal = np.asarray(x_normal)
        y_normal = np.asarray(y_normal)
        x_abnormal = np.asarray(x_abnormal)
        y_abnormal = np.asarray(y_abnormal)
        # print(x_normal.shape)
        # print(x_abnormal.shape)
        # for penalty term adjust training set only.
        # Normal: 1, Abnormal: 0
        # for 2040 normal data, we give 1200 for training and 840 for testing.
        self.x_tr, self.y_tr = x_normal[:1200], y_normal[:1200]
        self.x_te, self.y_te = x_normal[1200:], y_normal[1200:]
        # for 4294 abnormal data, we give 3300 for training and 994 for testing.
        print(self.x_tr.shape)
        print(x_abnormal.shape)
        self.x_tr = np.append(self.x_tr, x_abnormal[:3300], axis=0)
        self.y_tr = np.append(self.y_tr, y_abnormal[:3300], axis=0)

        self.x_te = np.append(self.x_te, x_abnormal[3300:], axis=0)
        self.y_te = np.append(self.y_te, y_abnormal[3300:], axis=0)

        print(self.x_tr.shape, self.y_tr.shape)
        print(self.x_te.shape, self.y_te.shape)

        self.x_tr, self.y_tr = shuffle(self.x_tr, self.y_tr)

    def load_data(self):
        path = "./cropped_512"
        paths = glob(os.path.join(path, '*.jpg'))
        # print(len(paths))
        return paths



    def get_other(self):
      np_path = "/content/drive/MyDrive/Colab Notebooks/DGM-TF/Data/nopneumonia/*.jpg"
      p_path = "/content/drive/MyDrive/Colab Notebooks/DGM-TF/Data/pneumonia/*.jpg"

      nps = glob(np_path)
      ps = glob(p_path)

      # print(len(nps), len(ps))

      normals = np.zeros(shape=[20672, 224, 224])
      n_labels = np.zeros(shape=[20672])

      abnormals = np.zeros(shape=[6012, 224, 224])
      ab_labels = np.zeros(shape=[6012])

      for i, d in enumerate(nps):
        img = cv2.imread(d, 0)
        img = cv2.resize(img, (224, 224))
        normals[i] = img
        n_labels[i] = 1
      # print(normals.dtype)
      for i, d in enumerate(ps):
        img = cv2.imread(d, 0)
        img = cv2.resize(img, (224, 224))
        abnormals[i] = img
        ab_labels[i] = 0
      
      normals = np.asarray(normals, np.float32)
      n_labels = np.asarray(n_labels, np.float32)
      abnormals = np.asarray(abnormals, np.float32)
      ab_labels = np.asarray(ab_labels, np.float32)
      print(normals.dtype)

      return normals, n_labels, abnormals, ab_labels





    def reset_idx(self): self.idx_tr, self.idx_te = 0, 0

    def next_train(self, batch_size=1, fix=False):

        start, end = self.idx_tr, self.idx_tr+batch_size  # !!!!
        x_tr, y_tr = self.x_tr[start:end], self.y_tr[start:end]
        x_tr = np.expand_dims(x_tr, axis=3)

        terminator = False
        if(end >= self.num_tr):
            terminator = True
            self.idx_tr = 0
            self.x_tr, self.y_tr = shuffle(self.x_tr, self.y_tr)
        else: self.idx_tr = end

        if(fix): self.idx_tr = start

        if(x_tr.shape[0] != batch_size):
            x_tr, y_tr = self.x_tr[-1-batch_size:-1], self.y_tr[-1-batch_size:-1]
            x_tr = np.expand_dims(x_tr, axis=3)

        if(self.normalize):
            min_x, max_x = x_tr.min(), x_tr.max()
            x_tr = (x_tr - min_x) / (max_x - min_x)

        return x_tr, y_tr, terminator

    def next_test(self, batch_size=1):

        start, end = self.idx_te, self.idx_te+batch_size  # !!!!
        x_te, y_te = self.x_te[start:end], self.y_te[start:end]
        x_te = np.expand_dims(x_te, axis=3)

        terminator = False
        if(end >= self.num_te):
            terminator = True
            self.idx_te = 0
        else: self.idx_te = end

        if(self.normalize):
            min_x, max_x = x_te.min(), x_te.max()
            x_te = (x_te - min_x) / (max_x - min_x)

        return x_te, y_te, terminator

dataset = Dataset()