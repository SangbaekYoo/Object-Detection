import cv2 as cv
from utils.global_variables import new_size, grid, img_w, img_h, segment, confindency_threshold

blue = (255, 0, 0)
green = (0, 255, 0)
red = (0, 0, 255)
white = (255, 255, 255)

def img_processing(src, vector, original=False):
<<<<<<< HEAD
=======
    print("entered")
>>>>>>> 8ecf968... renewal
    idx_grid_x, idx_grid_y, grid_x, grid_y = 0, 0, 0, 0
    x, y, w, h = 0, 0, 0, 0
    max_c = 0
    for i in range(4, len(vector), 5):
        idx_grid_x = (i % (5 * grid)) // 5
        idx_grid_y = i // (5 * grid)
        if vector[i] > max_c:
            max_c = vector[i]
            grid_x = idx_grid_x
            grid_y = idx_grid_y
            x, y, w, h = vector[i-4], vector[i-3], vector[i-2], vector[i-1]
    if confindency_threshold < max_c:
        x = int(x * segment) + grid_x * segment
        y = int(y * segment) + grid_y * segment
        w = int(w * new_size)
        h = int(h * new_size)
        if original:
            x = int(x * img_w / new_size)
            y = int(x * img_h / new_size)
            w = int(w * img_w / new_size)
            h = int(h * img_h / new_size)
        img = cv.rectangle(src, (x, y), (x + w, y + h), blue, 3)
    else:
        img = src
    cv.imshow('object detection', img)
    cv.waitKey(0)
    cv.destroyAllWindows()
<<<<<<< HEAD
=======
    return
>>>>>>> 8ecf968... renewal



