import cv2
import numpy as np

# get color from image CC1 or CC2
# out = ndarray[24:3] - масств цветов
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
# out = ndarray[400,500,3] - саму фотку CC
def generate_CC(colors, name, phase = 0, *tamplate):

    if phase == 0:
        output = np.empty((600, 900, 3), dtype='uint8')  # create empty mass for CC
        num_color = 0  # count
        for i in range(0, 600, 150):        # all vertical size, step
            for j in range(0, 900, 150):    # all horizontal size, step
                output[i:i + 100, j:j + 75, 0] = colors[num_color][0]  # set b channel
                output[i:i + 100, j:j + 75, 1] = colors[num_color][1]  # set g channel
                output[i:i + 100, j:j + 75, 2] = colors[num_color][2]  # set r channel
                num_color += 1
        #output = cv2.resize(output, (500,400))
        cv2.imshow(name, output)        # show result
        cv2.waitKey(0)
        return output
    if phase == 1:
        output = tamplate[0]
        num_color = 0  # count
        for i in range(0, 600, 150):        # all vertical size, step
            for j in range(0, 900, 150):    # all horizontal size, step
                output[i:i + 100,j+75:j + 150, 0] = colors[num_color][0]  # set b channel
                output[i:i + 100,j+75:j + 150, 1] = colors[num_color][1]  # set g channel
                output[i:i + 100,j+75:j + 150, 2] = colors[num_color][2]  # set r channel
                num_color += 1
        cv2.imshow(name, output)        # show result
        cv2.waitKey(0)
    if phase == 2:
        output = tamplate[0]
        num_color = 0  # count
        for i in range(0, 600, 150):        # all vertical size, step
            for j in range(0, 900, 150):    # all horizontal size, step
                output[i+100:i + 150,j:j + 150, 0] = colors[num_color][0]  # set b channel
                output[i+100:i + 150,j:j + 150, 1]= colors[num_color][1]  # set g channel
                output[i+100:i + 150,j:j + 150, 2] = colors[num_color][2]  # set r channel
                num_color += 1
        cv2.imshow(name, output)        # show result
        cv2.waitKey(0)
    return output

# distortion the color (s_color)
# out = ndarray [24,3] - значения шума
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
# mode 1 -> F = [1, b, g, r, bg, br, gr, rr, gg, bb]
# mode 2 -> F = [1, b, g, r, bg^0.5, br^0.5, gr^0.5]
def restruction(s_color, d_clolor, mode = 0):
    '''
    :param s_color: start color checker
    :param d_clolor: distortion color checker
    :param mode: 0 - I polin, 1 - II polin, 2 - root-polin
    :return:
    '''
    delta = d_clolor-s_color        # create deviation of colors
    dr = delta[:, 2].copy() // 1    # create deviation R channel
    dg = delta[:, 1].copy() // 1    # create deviation G channel
    db = delta[:, 0].copy() // 1    # create deviation B channel
    ones = np.ones((24, 1))
    # create other matrix (F)
    if mode == 0: F = np.concatenate([ones, s_color], axis=1)
    # полином 1 степени
    elif mode == 1:
        Fbg = (s_color[:, 2] * s_color[:, 1]).reshape(24, 1)
        Fbr = (s_color[:, 2] * s_color[:, 0]).reshape(24, 1)
        Fgr = (s_color[:, 1] * s_color[:, 0]).reshape(24, 1)
        Frr = (s_color[:, 0]).reshape(24, 1)**2
        Fgg = (s_color[:, 1]).reshape(24, 1) ** 2
        Fbb = (s_color[:, 2]).reshape(24, 1) ** 2
        F = np.hstack([ones, s_color, Fbg,  Fbr, Fgr, Frr, Fgg, Fbb])  # добавил одну степень полинома
    # полином 3 степени
    if mode == 2:
        Fbg = (s_color[:, 2] * s_color[:, 1]).reshape(24, 1)
        Fbr = (s_color[:, 2] * s_color[:, 0]).reshape(24, 1)
        Fgr = (s_color[:, 1] * s_color[:, 0]).reshape(24, 1)
        Frr = (s_color[:, 0]).reshape(24, 1) ** 2
        Fgg = (s_color[:, 1]).reshape(24, 1) ** 2
        Fbb = (s_color[:, 2]).reshape(24, 1) ** 2
        Frrr = (s_color[:, 0]).reshape(24, 1) ** 3
        Fggg = (s_color[:, 1]).reshape(24, 1) ** 3
        Fbbb = (s_color[:, 2]).reshape(24, 1) ** 3
        F = np.hstack([ones, s_color, Fbg, Fbr, Fgr, Frr, Fgg, Fbb, Frrr, Fggg, Fbbb])  # добавил одну степень полинома
    # полином 5 степени
    if mode == 3:
        Fbg = (s_color[:, 2] * s_color[:, 1]).reshape(24, 1)
        Fbr = (s_color[:, 2] * s_color[:, 0]).reshape(24, 1)
        Fgr = (s_color[:, 1] * s_color[:, 0]).reshape(24, 1)
        Frr = (s_color[:, 0]).reshape(24, 1) ** 2
        Fgg = (s_color[:, 1]).reshape(24, 1) ** 2
        Fbb = (s_color[:, 2]).reshape(24, 1) ** 2
        Frrr = (s_color[:, 0]).reshape(24, 1) ** 3
        Fggg = (s_color[:, 1]).reshape(24, 1) ** 3
        Fbbb = (s_color[:, 2]).reshape(24, 1) ** 3
        Frrrr = (s_color[:, 0]).reshape(24, 1) ** 4
        Fgggg = (s_color[:, 1]).reshape(24, 1) ** 4
        Fbbbb= (s_color[:, 2]).reshape(24, 1) ** 4
        Frrrrr = (s_color[:, 0]).reshape(24, 1) ** 5
        Fggggg = (s_color[:, 1]).reshape(24, 1) ** 5
        Fbbbbb = (s_color[:, 2]).reshape(24, 1) ** 5
        F = np.hstack([ones, s_color, Fbg, Fbr, Fgr, Frr, Fgg, Fbb, Frrr, Fggg, Fbbb,Frrrr, Fgggg, Fbbbb,
                       Frrrrr,Fggggg, Fbbbbb])  # добавил одну степень полинома
    elif mode == 4:
        Fbg = (s_color[:, 2] * s_color[:, 1]).reshape(24, 1) ** 0.5
        Fbr = (s_color[:, 2] * s_color[:, 0]).reshape(24, 1) ** 0.5
        Fgr = (s_color[:, 1] * s_color[:, 0]).reshape(24, 1) ** 0.5
        F = np.hstack([ones, s_color, Fbg, Fbr, Fgr])
    elif mode == 5:
        Fbg = (s_color[:, 2] * s_color[:, 1]).reshape(24, 1) ** 0.5
        Fbr = (s_color[:, 2] * s_color[:, 0]).reshape(24, 1) ** 0.5
        Fgr = (s_color[:, 1] * s_color[:, 0]).reshape(24, 1) ** 0.5
        Fbbg = (s_color[:, 2] * s_color[:, 2] * s_color[:, 1]).reshape(24, 1) ** (1/3)
        Fbbr = (s_color[:, 2] * s_color[:, 2] * s_color[:, 0]).reshape(24, 1) ** (1 / 3)
        Fggb = (s_color[:, 1] * s_color[:, 1] * s_color[:, 2]).reshape(24, 1) ** (1 / 3)
        Fggr = (s_color[:, 1] * s_color[:, 1] * s_color[:, 1]).reshape(24, 1) ** (1 / 3)
        Frrg = (s_color[:, 0] * s_color[:, 0] * s_color[:, 1]).reshape(24, 1) ** (1 / 3)
        Frrb = (s_color[:, 0] * s_color[:, 0] * s_color[:, 2]).reshape(24, 1) ** (1 / 3)
        F = np.hstack([ones, s_color, Fbg, Fbr, Fgr, Fbbg, Fbbr, Fggb, Fggr, Frrg, Frrb])
    Ftr = F.transpose()  # transpone F
    A = np.matmul(Ftr, F)
    A = np.linalg.inv(A)
    A = np.matmul(A, Ftr)  # get (Ft*F)^-1*Ft

    Ar = np.matmul(A, dr)  # find Ar
    Ag = np.matmul(A, dg)  # find Ag
    Ab = np.matmul(A, db)  # find Ab
    return Ab, Ag, Ar, F

# получаем значения скорректированной картинки
# out = ndarray[24,3]
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
result = generate_CC(start_colors, 'Start CC', 0)  # generate CC

noise3 = np.random.normal(4, 50, 3)  # main, SKO, numbers (B,G,R)
noise = np.full((24,3), noise3)      # double noise3 for 24 colors
#noise = np.random.normal(0, 20, (24,3))  # main, SKO, numbers (24 * B,G,R)

dist_colors = distortion_np(start_colors, noise)     # distortion colors
generate_CC(dist_colors, 'Distortion CC', 1, result)  # generate dist_CC

#for i in range(24):
#    print(start_colors[i], dist_color[i], noise[i]//1)      # start + noise = distortion
for m in range(6):

    Ab, Ag, Ar, F = restruction(start_colors, dist_colors, m)

    devR = np.matmul(F, Ar)          # get deviation Red
    devG = np.matmul(F, Ag)          # get deviation Green
    devB = np.matmul(F, Ab)          # get deviation Blue
    del_colors = np.vstack([devB.copy(), devG.copy(), devR.copy()]).transpose()     # create mass of deviation
    res_colors = get_result_colors(dist_colors, del_colors)
    generate_CC(res_colors, 'Restruction CC_'+str(m), 2, result)      # generate result ColorCheker
    delta = start_colors-res_colors
    delta_list = np.empty((24,3), dtype='uint16')
    for i in range(len(delta)):
        for j in range(len(delta[i])):
            if delta[i][j] > 200: delta_list[i][j] = 255-delta[i][j]
            else: delta_list[i][j] = delta[i][j]
    print(delta_list.mean(), delta_list.var())
    """print('Start colors, Generated noise, Distortion colors, Deviation of colors, Result colors mode = '+ str(m))
    for i in range(24):
        print(start_colors[i], noise[i] // 1, dist_colors[i] // 1, del_colors[i] // 1, res_colors[i] // 1, delta_list[i])
"""
