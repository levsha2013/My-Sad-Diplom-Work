import cv2
import numpy as np
import random

# искажение
def gasuss_noise(img=0):
    noise = np.random.normal(5, 0.5, 500)  # main, SKO, numbers
    v = np.var(noise)  # dispers
    m = np.mean(noise)
    print(m, v, "\n")

    return noise

img1 = cv2.imread('CC2.jpg')

img1 = cv2.resize(img1,(img1.shape[1]//2, img1.shape[0]//2))    # уменьшаю, чтобы влезло в экран
img = img1[:472,:]      # убираю надпись снизу


cv2.imshow('1 image', img1)
cv2.imshow('2 image', img)



print(img1.shape)
cv2.waitKey(0)

