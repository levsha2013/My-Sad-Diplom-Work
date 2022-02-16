import cv2
import numpy as np


# get color from image CC1 or CC2
def get_color_from_CC(img):
    output = []
    if img.shape[1] == 1200:
        for j in range(4):
            for i in range(6):
                output.append(img[200 + 180 * j][200 + 180 * i])
    else:
        for j in range(4):
            for i in range(6):
                output.append(img[150 + 223 * j][150 + 223 * i])
    return output

# generate CC
def generate_CC(colors, mode = 0):
    output = np.full((600, 900, 3), (0, 0, 0), dtype='uint8')  # create color
    num_color = 0
    for i in range(0, 600, 150):
        for j in range(0, 900, 150):
            output[i:i + 150, j:j + 150, 0] = colors[num_color][0]
            output[i:i + 150, j:j + 150, 1] = colors[num_color][1]
            output[i:i + 150, j:j + 150, 2] = colors[num_color][2]
            num_color += 1
    if mode == 0: cv2.imshow('CC', output)
    if mode == 1: cv2.imshow('CC_distortion', output)
    cv2.waitKey(0)
    return output

# incorrect distortion
def distortion(f_color, m = 0, d = 5):
    noise = np.random.normal(m, d, 24 * 3)  # main, SKO, numbers (24 colors B,G,R)
    n_color = []        # sample for new distorted colors
    for i in range(24):     # 24 colors
        n_color.append((np.ndarray((3), dtype='uint8')))    # add a new color
        for j in range(3):      # 3 channels
            if int(f_color[i][j]) + int(noise[i * 3]) > 255: n_color[i][j] = 255    # if sum more than 255
            elif int(f_color[i][j]) + int(noise[i * 3]) <0: n_color[i][j] = 0       # if sum less than 0
            else: n_color[i][j] = f_color[i][j] + noise[i * 3+j]                    # other cases
    return n_color


CC1 = cv2.imread('CC1.jpg')  # с цифрами
CC2 = cv2.imread('CC2.jpg')  # без цифр

f_color = get_color_from_CC(CC2)  # цвет, который ищем
generate_CC(f_color)  # generate CC

m = 0
disp = 50
n_color = distortion(f_color, m, disp)
generate_CC(n_color,1)  # generate distortion CC (mode = 1)