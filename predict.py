import cv2 as cv
import numpy as np
import pandas as pd
import tensorflow as tf

# Import module
from yolo.module import Module

import random

# Import global variables
from utils.global_variables import new_size, grid

# Import image process function
from utils.image_processing import img_processing

end = 2073

if __name__ == '__main__':
<<<<<<< HEAD
    module = Module()
    while True:
        idx = random.randint(0, end)
        img = cv.imread(f'./dataset/trainset/{idx}.png')
=======
    model = tf.keras.models.load_model('./ODmodel')
    while True:
        idx = random.randint(0, end)
        img_path = f'./dataset/trainset/{idx}.png'
        img_path.replace('/','\\\\')
        img = cv.imread(img_path)
>>>>>>> 8ecf968... renewal
        img_compressed = cv.resize(img, (new_size, new_size), interpolation=cv.INTER_AREA)
        pix = np.array(img_compressed)
        sentence = input('Continue?')
        if sentence == 'no':
            break
<<<<<<< HEAD
        pred = module.predict(pix / 255)
    pred_numpy = pred.numpy()
    img_processing(img, pred_numpy, original=True)
=======
        pred = model.predict(np.array([pix / 255]))
        img_processing(img_compressed, pred, original=False)
>>>>>>> 8ecf968... renewal



