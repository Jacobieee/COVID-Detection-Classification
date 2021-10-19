import numpy as np
import tensorflow as tf
import cv2
import tensorflow_addons as tfa

from sklearn.utils import shuffle

class Dataset(object):

    def __init__(self, normalize=True):

        print("\nInitializing Dataset...")

        self.normalize = False

        # (x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.mnist.load_data()

        # with np.load("train.npz") as data:
        #   a = data["x_train"]
        #   print(a)
          
        # read data.
        data_train = np.load("train_rsna.npz")
        data_test = np.load("test.npz")
        x_tr, y_tr = data_train["x_train"], data_train["y_train"]
        x_te, y_te = data_test["x_test"], data_test["y_test"]

        y_trnew = []
        for i, y in enumerate(y_tr):
          y_tmp = np.expand_dims(y_tr[i], axis=0)
          y_trnew.append(y_tmp)
        y_trnew = np.asarray(y_trnew, np.float32)
        assert(y_trnew.shape[0] == 26684)
        # for i, img in enumerate(x_tr):
        #   img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #   clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        #   x_tr[i] = clahe.apply(img)
        
        # for i, img in enumerate(x_te):
        #   img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #   clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        #   x_te[i] = clahe.apply(img)
        # for i, img in enumerate(x_tr):
        #   mask = self.get_mask(img)
        #   img_out = mask + img
        #   x_tr[i] = img_out
        
        # for i, img in enumerate(x_te):
        #   mask = self.get_mask(img)
        #   img_out = mask + img
        #   x_te[i] = img_out


        self.x_tr, self.y_tr = x_tr, y_trnew
        self.x_te, self.y_te = x_te, y_te
        print(f"training samples: {self.x_tr.shape}")
        # self.x_tr = np.ndarray.astype(self.x_tr, np.float32)
        # self.x_te = np.ndarray.astype(self.x_te, np.float32)

        # self.split_dataset()

        self.num_tr, self.num_te = self.x_tr.shape[0], self.x_te.shape[0]
        self.idx_tr, self.idx_te = 0, 0
        self.idx_normal = 0

        print("Number of data\nTraining: %d, Test: %d\n" %(self.num_tr, self.num_te))

        x_sample, y_sample = self.x_tr[0], self.y_tr[0]
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


    def sp_noise(self, image, prob):
      output = np.zeros(image.shape, np.uint8)
      thres = 1 - prob
      for i in range(image.shape[0]):
          for j in range(image.shape[1]):
              if image[i][j] < 15:
                  rdn = random.random()
                  if rdn < prob:
                      output[i][j] = 0
                  elif rdn > thres:
                      output[i][j] = 255
                  else:
                      output[i][j] = image[i][j]

      return output


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

        x_tot = np.append(self.x_tr, self.x_te, axis=0)
        y_tot = np.append(self.y_tr, self.y_te, axis=0)

        x_normal, y_normal = [], []
        x_abnormal, y_abnormal = [], []
        for yidx, y in enumerate(y_tot):

            y_tmp = np.expand_dims(y_tot[yidx], axis=0)

            if(y == 1): # as normal
                x_normal.append(x_tot[yidx])
                y_normal.append(y_tmp)
            else: # as abnormal
                x_abnormal.append(x_tot[yidx])
                y_abnormal.append(y_tmp)

            if((len(x_normal) >= 2000) and len(x_abnormal) >= 2000): break

        x_normal, y_normal = x_normal[:2000], y_normal[:2000]
        x_abnormal, y_abnormal = x_abnormal[:2000], y_abnormal[:2000]

        x_normal = np.asarray(x_normal)
        y_normal = np.asarray(y_normal)
        x_abnormal = np.asarray(x_abnormal)
        y_abnormal = np.asarray(y_abnormal)

        # for panalty term adjust training set only.
        # Normal: 1, Abnormal: 0
        self.x_tr, self.y_tr = x_normal[:1000], (y_normal[:1000] * 0) + 1
        self.x_te, self.y_te = x_normal[1000:], y_normal[1000:]

        self.x_tr = np.append(self.x_tr, x_abnormal[:1000], axis=0)
        self.y_tr = np.append(self.y_tr, (y_abnormal[:1000] * 0), axis=0)

        self.x_te = np.append(self.x_te, x_abnormal[1000:], axis=0)
        self.y_te = np.append(self.y_te, y_abnormal[1000:], axis=0)

        self.x_tr, self.y_tr = shuffle(self.x_tr, self.y_tr)

    def reset_idx(self): self.idx_tr, self.idx_te = 0, 0

    def next_train(self, batch_size=1, fix=False):

        start, end = self.idx_tr, self.idx_tr+batch_size
        start_normal, start_end = self.idx_normal, self.idx_normal+batch_size
        x_tr, y_tr = self.x_tr[start:end].copy(), self.y_tr[start:end].copy()

        x_tr = np.expand_dims(x_tr, axis=3)


        count = 0
        x_normals = np.zeros([batch_size, 224, 224])
        # y_normals = np.zeros([batch_size, 1])
        while count < batch_size:
          if self.idx_normal >= self.num_tr:
            self.idx_normal = 0
          if self.y_tr[self.idx_normal] == 1:
            x_normals[count] = self.x_tr[self.idx_normal].copy()
            # y_normals[count] = self.y_tr[self.idx_normal].copy()
            count += 1
          
          self.idx_normal += 1

        x_normals = np.expand_dims(x_normals, axis=3)

        terminator = False
        if(end >= self.num_tr):
            terminator = True
            self.idx_tr = 0
            self.x_tr, self.y_tr = shuffle(self.x_tr, self.y_tr)
        else: self.idx_tr = end

        if(fix): self.idx_tr = start

        if(x_tr.shape[0] != batch_size):
            x_tr, y_tr = self.x_tr[-1-batch_size:-1].copy(), self.y_tr[-1-batch_size:-1].copy()
            x_tr = np.expand_dims(x_tr, axis=3)

        if(self.normalize):
            min_x, max_x = x_tr.min(), x_tr.max()
            x_tr = (x_tr - min_x) / (max_x - min_x)

        for i, x in enumerate(x_tr):
          x_tr[i] = (x_tr[i] - x_tr[i].min())/(x_tr[i].max() - x_tr[i].min())
          x_tr[i] = 2*x_tr[i] - 1

        for i, x in enumerate(x_normals):
          x_normals[i] = (x_normals[i] - x_normals[i].min())/(x_normals[i].max() - x_normals[i].min())
          x_normals[i] = 2*x_normals[i] - 1

        return x_tr, y_tr, terminator, x_normals

    def next_test(self, batch_size=1):

        start, end = self.idx_te, self.idx_te+batch_size
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

# data = Dataset()