import cv2
import tensorflow as tf
import numpy as np

################################ This ones are not yet tested #######################################

class Dataset:

    def __init__(self):
        self.load_data()


    def load_data(self, file_name: str):
        f = open(file_name, 'r')
        lines = f.readlines()
        self.data = []
        for line in lines:
            line = line.strip()
            toks = line.split(',')
            path, cls_label, reg_label = toks[0], int(toks[1]), int(toks[2])
            self.data.append((path, cls_label, reg_label))


    def __getitem__(self, i):
        path, cls_label, reg_label = self.data[i]
        image = cv2.imread(path)
        return image, cls_label, reg_label

    
    def __len__(self):
        return len(self.data)
        



class DataLoader(tf.keras.utils.Sequence):
    def __init__(self, dataset: Dataset, batch_size = 1, shuffle = True,  what_to_do=0):
        self.ds = dataset 
        self.batch_size = batch_size 
        self.indices = np.arange(len(dataset))
        self.shuffle = shuffle
        #what_to_do = 0 <- classification, 1<-regression, 2<-both
        self.what_to_do = what_to_do
        self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle:
            self.indices = np.random.permutation(self.indices)

    def __len__(self):
        return len(self.indices)//self.batch_size

    def __getitem__(self, index):
        
        start = index*self.batch_size
        end = (index+1)*self.batch_size
    
        x_batch = []
        y_batch_cls = []
        y_batch_reg = []
        for j in range(start, end):
            image, cls_label, reg_label = self.ds[j]
            x_batch.append(image)
            y_batch_cls.append(cls_label)
            y_batch_reg.append(reg_label)

        if self.what_to_do == 0:
           return np.array(x_batch), np.array(y_batch_cls)  
        elif self.what_to_do == 1:
            return np.array(x_batch), np.array(y_batch_reg)  
        else:
            return np.array(x_batch), [np.array(y_batch_cls), np.array(y_batch_reg)]  


