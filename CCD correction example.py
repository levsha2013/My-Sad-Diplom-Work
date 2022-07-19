import cv2
import numpy as np
import colour
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from functions import *

def get_F_polin_for_rest(color, mode = 0):
    num_colors = len(color)

    color[0] = int(color[0])
    color[1] = int(color[1])
    color[2] = int(color[2])
    if mode == 0: F = np.concatenate([color], axis=0, dtype='float64')
    # полином 1 степени

    if mode == 1:
        Fbg = (int(color[2]) * int(color[1]))
        Fbr = (int(color[2]) * int(color[0]))
        Fgr = (int(color[1]) * int(color[0]))
        Frr = (int(color[0])) ** 2
        Fgg = (int(color[1])) ** 2
        Fbb = (int(color[2])) ** 2
        F = np.hstack([ color, Fbg,  Fbr, Fgr, Frr, Fgg, Fbb])  # добавил одну степень полинома
    # полином 3 степени недоделанный
    if mode == 2:
        Fbg = (color[:, 2] * color[:, 1]).reshape(num_colors, 1)
        Fbr = (color[:, 2] * color[:, 0]).reshape(num_colors, 1)
        Fgr = (color[:, 1] * color[:, 0]).reshape(num_colors, 1)
        Frr = (color[:, 0]).reshape(num_colors, 1) ** 2
        Fgg = (color[:, 1]).reshape(num_colors, 1) ** 2
        Fbb = (color[:, 2]).reshape(num_colors, 1) ** 2

        Frrr = (color[:, 0]).reshape(num_colors, 1) ** 3
        Frrg = (color[:, 0]**2 * color[:, 1]).reshape(num_colors, 1)
        Frrb = (color[:, 0]**2 * color[:, 2]).reshape(num_colors, 1)
        Fggg = (color[:, 1]).reshape(num_colors, 1) ** 3
        Fggr = (color[:, 1]**2 * color[:, 2]).reshape(num_colors, 1)
        Fggb = (color[:, 1] ** 2 * color[:, 0]).reshape(num_colors, 1)
        Fbbb = (color[:, 2]).reshape(num_colors, 1) ** 3
        Fbbr = (color[:, 2]**2 * color[:, 0]).reshape(num_colors, 1)
        Fbbg = (color[:, 2] ** 2 * color[:, 1]).reshape(num_colors, 1)
        Frgb = (color[:, 2] * color[:, 1] * color[:, 0]).reshape(num_colors, 1)
        F = np.hstack([color, Fbg, Fbr, Fgr, Frr, Fgg, Fbb, Frrr, Frrg, Frrb,
                       Fggg, Fggr, Fggb, Fbbb, Fbbr, Fbbg, Frgb])  # добавил одну степень полинома
    elif mode == 3:
        # root 1
        Fbg = (int(color[2]) * int(color[1])) ** 0.5
        Fbr = (int(color[2]) * int(color[0])) ** 0.5
        Fgr = (int(color[1]) * int(color[0])) ** 0.5
        F = np.hstack([ color, Fbg, Fbr, Fgr])
    elif mode == 4:
        # root 1
        Fbg = (color[:, 2] * color[:, 1]).reshape(num_colors, 1) ** 0.5
        Fbr = (color[:, 2] * color[:, 0]).reshape(num_colors, 1) ** 0.5
        Fgr = (color[:, 1] * color[:, 0]).reshape(num_colors, 1) ** 0.5


        Frrg = (color[:, 0] ** 2 * color[:, 1]).reshape(num_colors, 1) ** (1/3)
        Frrb = (color[:, 0] ** 2 * color[:, 2]).reshape(num_colors, 1) ** (1/3)

        Fggr = (color[:, 1] ** 2 * color[:, 2]).reshape(num_colors, 1) ** (1/3)
        Fggb = (color[:, 1] ** 2 * color[:, 0]).reshape(num_colors, 1) ** (1/3)

        Fbbr = (color[:, 2] ** 2 * color[:, 0]).reshape(num_colors, 1) ** (1/3)
        Fbbg = (color[:, 2] ** 2 * color[:, 1]).reshape(num_colors, 1) ** (1/3)
        Frgb = (color[:, 2] * color[:, 1] * color[:, 0]).reshape(num_colors, 1) ** (1/3)

        F = np.hstack([ color, Fbg, Fbr, Fgr, Frrg, Frrb, Fggr, Fggb, Fbbr, Fbbg, Frgb])
    return F

def function(A__):
    A_b = np.array(A__[0::3])
    A_g = np.array(A__[1::3])
    A_r = np.array(A__[2::3])

    F_1 = get_F_polin(dist_colors, MODE)
    tr_colors_r = np.matmul(F_1,A_r)
    tr_colors_g = np.matmul(F_1,A_g)
    tr_colors_b = np.matmul(F_1,A_b)

    del_colors = np.vstack([tr_colors_b.copy(), tr_colors_g.copy(), tr_colors_r.copy()]).transpose()     # create mass of deviation
    tr_colors = del_colors.astype('uint8')

    tr_colors_lab = bgr_to_lab_my(tr_colors)
    a = colour.difference.delta_E_CIE2000(start_colors_lab, tr_colors_lab) # вектор ошибки по CIEDE
    CIE_res = 0
    for i in range(len(a)):
        if i in interesting_colors: CIE_res += y1*a[i]# Y1*A[I]
        else:                       CIE_res += x1*a[i]#
    CIE_res = CIE_res / len(start_colors_lab)
    return CIE_res

CC1 = cv2.imread('CC1.jpg')
CC3 = cv2.imread('CC3.jpg')
CC4 = cv2.imread('NoWB_236_CMOS.png')



start_colors = get_color_from_CC_np(CC1)
numbers = len(start_colors)


regress_mode = 1  # 1 - MNK,  2 - optimise
MODE = 1          # 0 - линейный, 1 - полин2, 2 - полин3 2, 3 -рут2, 4 - рут3
p = 0.5          # 1 - без веса, больше - больше веса важным цветам

# номера цветов, которые несут большую для медицины информацию
interesting_colors = (24,25,26,38,39,40,52,53,56,66,67,68,80,81,82,
                      87,88,89,90,91,92,93,94,95,96,
                      101,102,103,104,105,106,107,108,109,110,
                      115,116,117,118,119,120,121,122,123,124)

y1 = p
x1 = 1-p

# цвета, взятые с CCD изображения
dist_colors = np.array([[13,31,30],[40,91,83],[44,54,29],[17,45,26],[55,69,50],[58,117,47],
               [20,84,97], [49,37,24], [26,57,83],[24,27,27], [27,123,67],[24,122,110],
               [37,26,14],[23,72,28],[16,41,71],[31,160,127],[45,62,84],[52,64,24],
               [123,244,162],[101,188,123],[64,121,78],[36,70,45], [19,37,25], [5,11,7]])

start_colors_lab = bgr_to_lab_my(start_colors)
C2 = [0.76]*numbers
for i in range(len(C2)):
    if i in interesting_colors:
        C2[i] = y1
    else: C2[i] = x1
C3 = np.zeros((numbers,numbers))
for i in range(numbers):
    C3[i][i] = C2[i]

A_b, A_g, A_r, res_colors = restruction(start_colors, dist_colors, MODE, 1, C3)
A__f = []
for j in range(len(A_b)):
    A__f.append(A_b[j])
    A__f.append(A_g[j])
    A__f.append(A_r[j])
res_first = function(A__f)

# для МНК
if regress_mode == 1:
    a = evklid_l2_norma(start_colors, res_colors)
    B_mnk, G_mnk, R_mnk, opt_mnk = restruction_with_wight(start_colors, dist_colors, a, MODE)
    A__mnk = []
    for j in range(len(A_b)):
        A__mnk.append(B_mnk[j])
        A__mnk.append(G_mnk[j])
        A__mnk.append(R_mnk[j])
    res_mnk = function(A__mnk) # результат по CIEDE

    #res_mnk_ev = evklid1_l2_norma(start_colors, function_last(A__mnk))

    if res_mnk >= res_first:
        real_res_mnk = res_first
        for_opt = A__f
    else:
        real_res_mnk = res_mnk
        for_opt = A__mnk

    res_colors_mnk = function_last(A__mnk, dist_colors, MODE)
# для функции оптимизации
if regress_mode == 1:
    res = minimize(function, for_opt, method='nelder-mead', options={'xtol': 1e-8, 'disp': True})
    A_opt = res.x           #модель
    res_colors_Q = function_last(res.x, dist_colors, MODE)

CC5 = CC4.copy()
print(CC5.shape)
leng = int(len(res.x)/3)
for i in range(CC5.shape[0]):
    for j in range(CC5.shape[1]):
        F = get_F_polin_for_rest(CC5[i,j],MODE)

        B = np.matmul(F, res.x[::3])
        G = np.matmul(F, res.x[1::3])
        R = np.matmul(F, res.x[2::3])
        CC5[i, j, 0] = B
        CC5[i, j, 1] = G
        CC5[i, j, 2] = R
        if B > 255: CC5[i, j, 0] = 255
        if B < 0:   CC5[i, j, 0] = 0
        if G > 255: CC5[i, j, 1] = 255
        if G < 0:   CC5[i, j, 1] = 0
        if R > 255: CC5[i, j, 2] = 255
        if R < 0:   CC5[i, j, 2] = 0
# dsize
dsize = (int(CC4.shape[1] * 50 / 100), int(CC4.shape[0] * 70 / 100))

# resize image
cv2.imshow('input_img', cv2.resize(CC4, dsize))
cv2.imshow('output_img', cv2.resize(CC5, dsize))
cv2.waitKey()

generate_result(start_colors, dist_colors, function_last(res.x, dist_colors, MODE))
