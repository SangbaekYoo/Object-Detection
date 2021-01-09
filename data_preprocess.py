import PIL.Image as pilimg
import numpy as np
import pandas as pd

img_W = 1920
img_H = 1080

new_size = 400

grid = 4

segment = new_size // grid

def location_estimate(x, y):
    x_index, y_index = x // segment, y // segment
    x_coordinate, y_coordinate = x % segment, y % segment
    return x_index, y_index, x_coordinate, y_coordinate

img_data_dir_path = './dataset/trainset/'
label_data_path = './dataset/trainset_label.csv'

label_data = pd.read_csv(label_data_path)
data_numpy = label_data.to_numpy()

if __name__ == '__main__':
    pix_lib = np.empty((0, new_size, new_size, 3))
    numerical_lib = np.empty((0, (grid**2)*5))

    for element in data_numpy:
        img_name = element[0]
        img = pilimg.open(f'{img_data_dir_path}{img_name}')
        img = img.resize((new_size, new_size))
        pix = np.array(img)
        element_preprocess = np.zeros((grid, grid,5))
        numerical_data = element[1].split()
        x, y, w, h = int(numerical_data[0]), int(numerical_data[1]), int(numerical_data[2]), int(numerical_data[3])
        x = int(x * new_size / img_W)
        y = int(y * new_size / img_H)
        w = int(w * new_size / img_W)
        h = int(h * new_size / img_H)
        x_index, y_index, x_coordinate, y_coordinate = location_estimate(x, y)
        element_preprocess[x_index][y_index][0] = x_coordinate / segment
        element_preprocess[x_index][y_index][1] = y_coordinate / segment
        element_preprocess[x_index][y_index][2] = w / new_size
        element_preprocess[x_index][y_index][3] = h / new_size
        element_preprocess[x_index][y_index][4] = 1
        element_preprocess = element_preprocess.reshape((grid**2)*5)
        pix_lib = np.append(pix_lib, np.array([pix]), axis=0)
        numerical_lib = np.append(numerical_lib, np.array([element_preprocess]), axis=0)
        print(element[0])

    df_pix = pd.DataFrame(pix_lib)
    df_num = pd.DataFrame(numerical_lib)

    df_pix.to_csv('./dataset/pix_data.csv')
    df_num.to_csv('./dataset/num_data.csv')


