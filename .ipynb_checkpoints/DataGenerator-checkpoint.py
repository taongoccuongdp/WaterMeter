import json
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras import backend as K
import random
from random import randint
from PIL import Image
random.seed(2020)
class DataGenerator:
    def __init__(self, json_path, img_w, img_h, batch_size, train = True):
        self.train = train
        self.data = {}
        self.data = json.load(open(json_path)) if json_path != None else None
        self.img_w = img_w
        self.img_h = img_h
        self.batch_size = batch_size
        self.imgs = list(self.data.keys())
        random.shuffle(self.imgs)
        self.current_key = 0
        self.samples_size = len(self.imgs)
        if self.train == False:
            self.current_key = self.current_key + self.samples_size//5 * 4
    def next_sample(self):
        img = cv2.imread(self.imgs[self.current_key], cv2.IMREAD_GRAYSCALE)
        new_img = cv2.resize(img, (self.img_h, self.img_w))
        label = self.data[self.imgs[self.current_key]]
        self.current_key = self.current_key + 1
        if self.current_key > self.samples_size:
            self.current_key = 0  
            if self.train == False:
                self.current_key = self.current_key + self.samples_size//5 * 4
            random.shuffle(self.imgs)
        return new_img, label
    def next_batch(self):
        X_data = np.zeros([self.batch_size, self.img_w, self.img_h, 1], dtype=np.float32)
        Y_data = np.zeros([self.batch_size, 6], dtype=np.float32)
        input_length = np.ones([self.batch_size, 1], dtype=np.float32)
        label_length = np.ones([self.batch_size, 1], dtype=np.float32)
        for i in range(self.batch_size):
            img, label = self.next_sample()
            print(img.shape)
            X_data[i] = img
            spl = list(map(lambda x: float(x), label.split(',')))
            Y_data[i, :len(spl)] = spl
        return X_data, Y_data
        
        