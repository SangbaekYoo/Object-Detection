import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models


class Model_Based_YOLO(tf.keras.Model):
    def __init__(self, img_size, grid):
        super(Model_Based_YOLO, self).__init__()
        self.img_size = img_size
        self.grid = grid
        self.conv1 = layers.Conv2D(64, (self.img_size - 2 * self.grid + 1, self.img_size - 2 * self.grid + 1),
                                   activation='relu', padding='valid', input_shape=(self.img_size, self.img_size, 3))
        self.conv2 = layers.Conv2D(64, (3, 3),
                                   activation='relu', padding='same', input_shape=(2 * self.grid, 2 * self.grid, 64))
        self.conv3= layers.Conv2D(64, (self.grid + 1, self.grid + 1),
                                   activation='relu', padding='valid', input_shape=(2 * self.grid, 2 * self.grid, 64))
        self.conv4 = layers.Conv2D(64, (self.grid + 1, self.grid + 1),
                                   activation='relu', padding='valid', input_shape=(2 * self.grid, 2 * self.grid, 64))
        self.conv5 = layers.Conv2D(64, (3, 3),
                                   activation='relu', padding='same', input_shape=(self.grid, self.grid, 64))
        self.flatten = layers.Flatten()

        self.fc1 = layers.Dense(128, activation='relu')
        self.fc2 = layers.Dense((self.grid**2)*5)

    def call(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.conv4(y)
        y = self.conv5(y)
        y = self.flatten(y)
        y = self.fc1(y)
        output = self.fc2(y)
        #output = tf.reshape(output, )
        return output
