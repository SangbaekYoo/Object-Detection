import tensorflow as tf
import numpy as np
import cv2 as cv
import os

from utils.global_variables import new_size, grid, img_w, img_h, segment, confindency_threshold

data_path = __file__.replace('utils\image_data_extraction.py', '')

def imread(filename, flags=cv.IMREAD_COLOR, dtype=np.uint8):
    try:
        n = np.fromfile(filename, dtype)
        img = cv.imdecode(n, flags)
        return img
    except Exception as e:
        print(e)
        return None

def img_extraction(label_batch_data):
    pix_lib = np.empty((0, new_size, new_size, 3))
    batch_data = label_batch_data.numpy()
    for label in batch_data:
        file_path = f'{data_path}dataset\\trainset\\{label[0]}.png'
        img = imread(file_path)
        img_compressed = cv.resize(img, (new_size, new_size), cv.INTER_AREA)
        pix_lib = np.append(pix_lib, np.array([img_compressed / 255]), axis=0)
    return tf.convert_to_tensor(pix_lib)

if __name__ == '__main__':
    batch_data = tf.convert_to_tensor([0,1,2,3,4,5,6])