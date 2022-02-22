import cv2
import numpy as np

# get color from image CC1 or CC2
def get_color_from_CC_np(img):
    output = np.empty((24,3), dtype='uint8')       # create empty mass for colors
    if img.shape[1] == 1200:        # if CC = CC1
        for i in range(4):
            for j in range(6):
                output[i*6+j] = img[200 + 180 * i, 200 + 180 * j]   # take a color
    else:                           # if CC  = CC2
        for i in range(4):
            for j in range(6):
                output[i*6+j] = img[150 + 223 * i, 150 + 223 * j]   # take a color
    return output

# generate CC and show it
def generate_CC(colors, name):
    output = np.empty((600, 900, 3), dtype='uint8')  # create empty mass for CC
    num_color = 0       # count
    for i in range(0, 600, 150):        # all vertical size, step
        for j in range(0, 900, 150):    # all horizontal size, step
            output[i:i + 150, j:j + 150, 0] = colors[num_color][0]  # set b channel
            output[i:i + 150, j:j + 150, 1] = colors[num_color][1]  # set g channel
            output[i:i + 150, j:j + 150, 2] = colors[num_color][2]  # set r channel
            num_color += 1
    output = cv2.resize(output, (500,400))
    cv2.imshow(name, output)        # show result
    cv2.waitKey(0)
    return output

# distortion the color (s_color)
def distortion_np(s_color, noise):
    output = np.zeros((24,3))
    for i in range(24):     # 24 colors
        for j in range(3):      # 3 channels
            if s_color[i, j] + noise[i, j] > 255: output[i, j] = 255    # if sum more than 255
            elif s_color[i, j] + noise[i, j] < 0: output[i, j] = 0      # if sum less than 0
            else: output[i, j] = s_color[i, j] + noise[i, j]            # other cases
    return output

# create Ab, Ag, Ar for distortion colors
# mode 0 -> F = [1, b, g, r]
# mode 1 -> F = [1, b, g, r, bg, br, gr]
def restruction(s_color, d_clolor, mode = 0):
    delta = d_clolor-s_color        # create deviation of colors
    dr = delta[:, 2].copy() // 1    # create deviation R channel
    dg = delta[:, 1].copy() // 1    # create deviation G channel
    db = delta[:, 0].copy() // 1    # create deviation B channel
    ones = np.ones((24, 1))
    # create other matrix
    if mode == 0: F = np.concatenate([ones, s_color], axis=1)
    else:
        Fbg = (s_color[:, 2] * s_color[:, 1]).reshape(24, 1)
        Fbr = (s_color[:, 2] * s_color[:, 0]).reshape(24, 1)
        Fgr = (s_color[:, 1] * s_color[:, 0]).reshape(24, 1)
        F = np.hstack([ones, s_color, Fbg,  Fbr, Fgr])  # добавил одну степень полинома
    Ftr = F.transpose()  # transpone F
    A = np.matmul(Ftr, F)
    A = np.linalg.inv(A)
    A = np.matmul(A, Ftr)  # get (Ft*F)^-1*Ft

    Ar = np.matmul(A, dr)  # find Ar
    Ag = np.matmul(A, dg)  # find Ag
    Ab = np.matmul(A, db)  # find Ab
    return Ab, Ag, Ar, F


def get_result_colors(dist_color, del_color):
    output = np.empty((24,3), dtype='uint8')
    for i in range(24):
        for j in range(3):
            a = dist_color[i,j]-del_color[i,j]
            if a > 255:
                output[i,j] = 255
            elif a < 0.0: output[i,j] = 0
            else: output[i,j] = (dist_color[i,j]-del_color[i,j])//1
    return output


CC1 = cv2.imread('CC1.jpg')  # with values
CC2 = cv2.imread('CC2.jpg')  # without values

start_colors = get_color_from_CC_np(CC2)  # take the colors from CC
generate_CC(start_colors, 'Start CC')  # generate CC

noise3 = np.random.normal(0, 20, 3)  # main, SKO, numbers (B,G,R)
noise = np.full((24,3), noise3)      # double noise3 for 24 colors
#noise = np.random.normal(0, 20, (24,3))  # main, SKO, numbers (24 * B,G,R)

dist_colors = distortion_np(start_colors, noise)     # distortion colors
generate_CC(dist_colors, 'Distortion CC')  # generate dist_CC

#for i in range(24):
#    print(start_colors[i], dist_color[i], noise[i]//1)      # start + noise = distortion

Ab, Ag, Ar, F = restruction(start_colors, dist_colors, 0)

devR = np.matmul(F, Ar)          # get deviation Red
devG = np.matmul(F, Ag)          # get deviation Green
devB = np.matmul(F, Ab)          # get deviation Blue
del_colors = np.vstack([devB.copy(), devG.copy(), devR.copy()]).transpose()     # create mass of deviation

res_colors = get_result_colors(dist_colors, del_colors)
generate_CC(res_colors, 'Restruction CC')      # generate result ColorCheker

print('Start colors, Generated noise, Distortion colors, Deviation of colors, Result colors')
for i in range(24):
    print(start_colors[i], noise[i], dist_colors[i], del_colors[i], res_colors[i])       # result colors and start colors
cv2.waitKey(0)
