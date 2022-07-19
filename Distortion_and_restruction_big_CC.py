import cv2
import colour
from scipy.optimize import minimize
from functions import *

def function(A__):
    A_b = np.array(A__[0::3])
    A_g = np.array(A__[1::3])
    A_r = np.array(A__[2::3])

    F_1 = get_F_polin(dist_colors, MODE)
    tr_colors_r = np.matmul(F_1, A_r)
    tr_colors_g = np.matmul(F_1, A_g)
    tr_colors_b = np.matmul(F_1, A_b)

    del_colors = np.vstack(
        [tr_colors_b.copy(), tr_colors_g.copy(), tr_colors_r.copy()]).transpose()  # create mass of deviation
    tr_colors = del_colors.astype('uint8')

    tr_colors_lab = bgr_to_lab_my(tr_colors)
    a = colour.difference.delta_E_CIE2000(start_colors_lab, tr_colors_lab)  # вектор ошибки по CIEDE
    CIE_res = 0
    for i in range(len(a)):
        if i in interesting_colors:
            CIE_res += p * a[i]
        else:
            CIE_res += 1/p * a[i]
    CIE_res = CIE_res / len(start_colors_lab)
    return CIE_res

CC1 = cv2.imread('CC1.jpg')
CC3 = cv2.imread('CC3.jpg')
start_colors = get_color_from_CC_np(CC3)
# start_colors = np.array([[50,60,70],[110,123,130],[80,90,100],])
# start_colors = np.random.randint(0, 255, (3,3))
numbers = len(start_colors)

regress_mode = 1  # 1 - MNK,  2 - optimise
MODE = 3  # 0 - линейный, 1 - полин2, 2 - полин3 2, 3 -рут2, 4 - рут3
p = 25  # 1 - без веса, больше - больше веса важным цветам
D = 5   # значение СКО


noise3 = np.random.normal(0, D, 1)
noise = np.full((len(start_colors), 3), noise3)
print(noise3)
interesting_colors = (24, 25, 26, 38, 39, 40, 52, 53, 56, 66, 67, 68, 80, 81, 82,
                      87, 88, 89, 90, 91, 92, 93, 94, 95, 96,
                      101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
                      115, 116, 117, 118, 119, 120, 121, 122, 123, 124)

x1 = 140 / (45 * p + 95)
y1 = (140 - 95 * x1) / 45
dist_colors = distortion_np(start_colors, noise)
start_colors_lab = bgr_to_lab_my(start_colors)
C2 = [0.76] * numbers

for i in range(len(C2)):
    if i in interesting_colors:  C2[i] = y1
    else:                        C2[i] = x1
C3 = np.zeros((numbers, numbers))
for i in range(numbers):         C3[i][i] = C2[i]


A_b, A_g, A_r, res_colors = restruction(start_colors, dist_colors, MODE, 1, C3)

# для функции оптимизации

A__f = []
for j in range(len(A_b)):
    A__f.append(A_b[j])
    A__f.append(A_g[j])
    A__f.append(A_r[j])

res_first = function(A__f)

res = minimize(function, A__f, method='nelder-mead', options={'xtol': 1e-8, 'disp': True})


M_opt = res.x       # итоговая модель (параметры)
res_colors = function_last(M_opt,dist_colors, MODE)
generate_result2(start_colors, dist_colors, res_colors)