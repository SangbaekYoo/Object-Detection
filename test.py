import PIL.Image as pilimg
import numpy as np
import pandas as pd
import pickle



pix_lib = np.zeros((5, 5, 5, 3), dtype=float)

np.savez('./x',x = pix_lib)

y = np.load('./x.npz')
print(type(y['x']))