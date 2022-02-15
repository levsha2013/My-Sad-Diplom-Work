import cv2
import numpy as np
import random

# distortion
def gasuss_noise(img=0):
    noise = np.random.normal(5, 0.5, 500)  # main, SKO, numbers
    v = np.var(noise)  # disper
    m = np.mean(noise)
    #print(m, v, "\n")
    return noise

# change one fix color
def one_fix_color_change():
    start = np.full((400, 400, 3), (60, 54, 175), dtype='uint8')  # create color
    noise = np.random.normal(0, 20, 3)  # main, SKO, numbers
    result = np.full((400, 400, 3), (60, 54, 175), dtype='uint8')  # create ndarray for change

    result[:, :, 2] = start[:, :, 2] + noise[2]  # change R
    result[:, :, 1] = start[:, :, 1] + noise[1]  # change G
    result[:, :, 0] = start[:, :, 0] + noise[0]  # change B

    print(start[0][0])  # start color
    print(noise)  # noise
    print(result[0][0])  # result color

    cv2.imshow('1', start)
    cv2.imshow('2', result)
    cv2.waitKey(0)

# get color from image CC1
def get_color_from_CC1(CC1):
    output = []
    for j in range(4):
        for i in range(6):
            output.append(CC1[200+180*j][200+180*i])
    return output

# get color from image CC2
def get_color_from_CC2(CC2):
    output = []
    for j in range(4):
        for i in range(6):
            output.append(CC2[150+223*j][150+223*i])
    return output

# find the same f_color(+-step) and change it
def try_1_to_change(img, f_color, step = 15):
    for j in range(img.shape[0]):
        for i in range(img.shape[1]):
            if(img[j][i][0] <= f_color[0]+step) and (img[j][i][0] >= f_color[0]-step) \
                    and (img[j][i][1] <= f_color[1]+step) and (img[j][i][1] >= f_color[1]-step)  \
                    and (img[j][i][2] <= f_color[2]+step) and (img[j][i][2] >= f_color[2]-step):
                img[j,i,0] = 0
                img[j,i,1] = 255
                img[j,i,2] = 255
    return img

# generate CC
def generate_CC(colors):
    output = np.full((600, 900, 3), (0, 0, 0), dtype='uint8')  # create color
    num_color = 0
    for i in range(0,600,150):
        for j in range(0,900,150):
            output[i:i + 150, j:j + 150, 0] = f_color[num_color][0]
            output[i:i + 150, j:j + 150, 1] = f_color[num_color][1]
            output[i:i + 150, j:j + 150, 2] = f_color[num_color][2]
            num_color+=1
    return output


CC1 = cv2.imread('CC1.jpg') # с цифрами
CC2 = cv2.imread('CC2.jpg') # без цифр


f_color = get_color_from_CC2(CC2)            # цвет, который ищем
output = generate_CC(f_color)
cv2.imshow('123',output)
cv2.waitKey(0)
cv2.destroyAllWindows()

f_color = get_color_from_CC1(CC1)            # цвет, который ищем
output = generate_CC(f_color)
cv2.imshow('123',output)
cv2.waitKey(0)