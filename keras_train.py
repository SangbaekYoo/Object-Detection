import tensorflow as tf
import numpy as np
from utils.global_variables import new_size, grid, img_w, img_h, segment, confindency_threshold


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D

if __name__ == '__main__':
    # Load train data
    pix_data = np.load('./dataset/pix_data.npz')['one'][:100]
    num_data = np.load('./dataset/numerical_data.npz')['one'][:100]

    print(len(pix_data))
    print(len(num_data))

    model = Sequential()
    model.add(Conv2D(64, (new_size - 2 * grid + 1, new_size - 2 * grid + 1),
                                   activation='relu', padding='valid', input_shape=(new_size, new_size, 3)))
    model.add(Conv2D(64, (3, 3),
                                   activation='relu', padding='same', input_shape=(2 * grid, 2 * grid, 64)))
    model.add(Conv2D(64, (grid + 1, grid + 1),
                                   activation='relu', padding='valid', input_shape=(2 * grid, 2 * grid, 64)))
    model.add(Conv2D(64, (3, 3),
                                   activation='relu', padding='same', input_shape=(grid, grid, 64)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense((grid**2)*5))

    model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
    model.fit(pix_data, num_data, batch_size=1, epochs=1, shuffle=True)

    model.save('./ODmodel')


