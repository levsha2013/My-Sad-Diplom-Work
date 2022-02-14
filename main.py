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


img1 = cv2.imread('CC2.jpg')

img1 = cv2.resize(img1,(img1.shape[1]//2, img1.shape[0]//2))    # уменьшаю, чтобы влезло в экран
img = img1[:472,:]      # убираю надпись снизу


#cv2.imshow('1 image', img1)
#cv2.imshow('2 image', img)

s = set()
d = dict()
for i in range(0,img1.shape[0],10):
    for j in range(0,img1.shape[1],10):
        #if not img1[i][j][0] in s:
            #s.add(img1[i][j][0])
        pass
#print(s)
#cv2.imshow('test', test_img)

# создаем цвет и искажаем его шумом
