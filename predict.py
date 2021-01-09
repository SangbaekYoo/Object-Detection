import PIL.Image as pilimg
import numpy as np
import pandas as pd

# Import module
from yolo.module import Module

import random

# Import global variables
from utils.global_variables import new_size, grid

end = 2073

if __name__ == '__main__':
    module = Module()
    idx = random.randint(0, end)

    img = pilimg.open(f'./dataset/trainset/{idx}.png')
    img = img.resize((new_size, new_size))
    pix = np.array(img)

    while True:
        sentence = input('Continue?')
        if sentence == 'no':
            break
        pred = module.predict(pix)

