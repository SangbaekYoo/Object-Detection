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
    module = Module()
    while True:
        idx = random.randint(0, end)
        img = cv.imread(f'./dataset/trainset/{idx}.png')
        img_compressed = cv.resize(img, (new_size, new_size), interpolation=cv.INTER_AREA)
        pix = np.array(img_compressed)
        sentence = input('Continue?')
        if sentence == 'no':
            break
        pred = module.predict(pix / 255)
    pred_numpy = pred.numpy()
    img_processing(img, pred_numpy, original=True)



