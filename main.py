import cv2

img = cv2.imread('CC2.jpg')
#cv2.imshow('Start image', img)
#cv2.waitKey(5000)

#mov = cv2.VideoCapture(0)                   # считываем видео с камеры (0)
mov = cv2.VideoCapture('video1.mp4')       # считываем видео с файла
mov.set(3,9000)                              # 3 - ширина, 4 - высота
mov.set(4,300)

success= 1
while(True):
    success, img = mov.read()
    cv2.imshow('Result', img)
    if (cv2.waitKey(50)) & 0xFF == ord('q'):
        break

cv2.waitKey(0)