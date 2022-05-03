import cv2
import numpy as np
import colour
import random as rand
# get color from image CC1 or CC2
# out = ndarray[24:3] - масств цветов

def rgb2lab ( inputColor ) :

   num = 0
   RGB = [0, 0, 0]

   for value in inputColor :
       value = float(value) / 255

       if value > 0.04045 :
           value = ( ( value + 0.055 ) / 1.055 ) ** 2.4
       else :
           value = value / 12.92

       RGB[num] = value * 100
       num = num + 1

   XYZ = [0, 0, 0,]

   X = RGB [0] * 0.4124 + RGB [1] * 0.3576 + RGB [2] * 0.1805
   Y = RGB [0] * 0.2126 + RGB [1] * 0.7152 + RGB [2] * 0.0722
   Z = RGB [0] * 0.0193 + RGB [1] * 0.1192 + RGB [2] * 0.9505
   XYZ[ 0 ] = round( X, 4 )
   XYZ[ 1 ] = round( Y, 4 )
   XYZ[ 2 ] = round( Z, 4 )

   XYZ[ 0 ] = float( XYZ[ 0 ] ) / 95.047         # ref_X =  95.047   Observer= 2°, Illuminant= D65
   XYZ[ 1 ] = float( XYZ[ 1 ] ) / 100.0          # ref_Y = 100.000
   XYZ[ 2 ] = float( XYZ[ 2 ] ) / 108.883        # ref_Z = 108.883

   num = 0
   for value in XYZ :

       if value > 0.008856 :
           value = value ** ( 0.3333333333333333 )
       else :
           value = ( 7.787 * value ) + ( 16 / 116 )

       XYZ[num] = value
       num = num + 1

   Lab = [0, 0, 0]

   L = ( 116 * XYZ[ 1 ] ) - 16
   a = 500 * ( XYZ[ 0 ] - XYZ[ 1 ] )
   b = 200 * ( XYZ[ 1 ] - XYZ[ 2 ] )

   Lab [ 0 ] = round( L, 4 )
   Lab [ 1 ] = round( a, 4 )
   Lab [ 2 ] = round( b, 4 )

   return Lab

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


def generate_result(color_1, color_2, color_3, name='result'):

    output = np.empty((600, 900, 3), dtype='uint8')  # create empty mass for CC
    num_color = 0  # count
    for i in range(0, 600, 150):        # all vertical size, step
        for j in range(0, 900, 150):    # all horizontal size, step
            output[i:i + 80, j:j + 75, 0] = color_1[num_color][0]  # set b channel
            output[i:i + 80, j:j + 75, 1] = color_1[num_color][1]  # set g channel
            output[i:i + 80, j:j + 75, 2] = color_1[num_color][2]  # set r channel

            output[i:i + 80, j + 75:j + 150, 0] = color_2[num_color][0]  # set b channel
            output[i:i + 80, j + 75:j + 150, 1] = color_2[num_color][1]  # set g channel
            output[i:i + 80, j + 75:j + 150, 2] = color_2[num_color][2]  # set r channel

            output[i + 80:i + 150, j:j + 150, 0] = color_3[num_color][0]  # set b channel
            output[i + 80:i + 150, j:j + 150, 1] = color_3[num_color][1]  # set g channel
            output[i + 80:i + 150, j:j + 150, 2] = color_3[num_color][2]  # set r channel

            num_color += 1
    cv2.imshow(name, output)        # show result
    cv2.waitKey(500)
    cv2.destroyWindow('result')


# distortion the color (s_color)
# out = ndarray [24,3] - значения шума
def distortion_np(s_color, noise):
    output = np.zeros((len(s_color),3))
    for i in range(len(s_color)):     # 24 colors
        for j in range(3):      # 3 channels
            if s_color[i, j] + noise[i, j] > 255: output[i, j] = 255    # if sum more than 255
            elif s_color[i, j] + noise[i, j] < 0: output[i, j] = 0      # if sum less than 0
            else: output[i, j] = s_color[i, j] + noise[i, j]            # other cases
    return output


    ''' В О С С Т А Н О В Л Е Н И Е'''
# create Ab, Ag, Ar for distortion colors
# mode 0 -> F = [1, b, g, r]
# mode 1 -> F = [1, b, g, r, bg, br, gr, rr, gg, bb]
# mode 2 -> F = [1, b, g, r, bg^0.5, br^0.5, gr^0.5]
def restruction(s_color, d_clolor, mode =0):
    '''
    :param s_color: start color checker
    :param d_clolor: distortion color checker
    :param mode: 0 - I polin, 1 - II polin, 2 - root-polin
    :return:
    '''
    num_colors = len(s_color)
    delta = d_clolor-s_color        # create deviation of colors
    dr = delta[:, 2].copy() // 1    # create deviation R channel
    dg = delta[:, 1].copy() // 1    # create deviation G channel
    db = delta[:, 0].copy() // 1    # create deviation B channel
    ones = np.ones((num_colors, 1))
    set_r = {"r", 'g', 'b'}
    for i in range(1):
        set_n = set()
        for j in set_r:
            r_new = "".join(sorted(j + 'r'))
            b_new = "".join(sorted(j + 'b'))
            g_new = "".join(sorted(j + 'g'))
            set_n.add(r_new)
            set_n.add(b_new)
            set_n.add(g_new)
        [set_r.add(j) for j in set_n]
    list_of_set = sorted(set_r)
    mode = 'polin'
    print(mode)
    result = []
    for i in list_of_set:
        if mode == 'polin':
            c = 1
            for j in range(len(i)):
                if i[j] == 'r': c = c * dr
                if i[j] == 'b': c = c * db
                if i[j] == 'g': c = c * dg
            result.append(c)
    if mode == 0: F = np.concatenate([ones, s_color], axis=1)
    # полином 1 степени
    if mode == 1:
        Fbg = (s_color[:, 2] * s_color[:, 1]).reshape(num_colors, 1)
        Fbr = (s_color[:, 2] * s_color[:, 0]).reshape(num_colors, 1)
        Fgr = (s_color[:, 1] * s_color[:, 0]).reshape(num_colors, 1)
        Frr = (s_color[:, 0]).reshape(num_colors, 1)**2
        Fgg = (s_color[:, 1]).reshape(num_colors, 1) ** 2
        Fbb = (s_color[:, 2]).reshape(num_colors, 1) ** 2
        F = np.hstack([ones, s_color, Fbg,  Fbr, Fgr, Frr, Fgg, Fbb])  # добавил одну степень полинома
    # полином 3 степени недоделанный
    if mode == 2:
        Fbg = (s_color[:, 2] * s_color[:, 1]).reshape(num_colors, 1)
        Fbr = (s_color[:, 2] * s_color[:, 0]).reshape(num_colors, 1)
        Fgr = (s_color[:, 1] * s_color[:, 0]).reshape(num_colors, 1)
        Frr = (s_color[:, 0]).reshape(num_colors, 1) ** 2
        Fgg = (s_color[:, 1]).reshape(num_colors, 1) ** 2
        Fbb = (s_color[:, 2]).reshape(num_colors, 1) ** 2

        Frrr = (s_color[:, 0]).reshape(num_colors, 1) ** 3
        Frrg = (s_color[:, 0]**2 * s_color[:, 1]).reshape(num_colors, 1)
        Frrb = (s_color[:, 0]**2 * s_color[:, 2]).reshape(num_colors, 1)
        Fggg = (s_color[:, 1]).reshape(num_colors, 1) ** 3
        Fggr = (s_color[:, 1]**2 * s_color[:, 2]).reshape(num_colors, 1)
        Fggb = (s_color[:, 1] ** 2 * s_color[:, 0]).reshape(num_colors, 1)
        Fbbb = (s_color[:, 2]).reshape(num_colors, 1) ** 3
        Fbbr = (s_color[:, 2]**2 * s_color[:, 0]).reshape(num_colors, 1)
        Fbbg = (s_color[:, 2] ** 2 * s_color[:, 1]).reshape(num_colors, 1)
        Frgb = (s_color[:, 2] * s_color[:, 1] * s_color[:, 0]).reshape(num_colors, 1)
        F = np.hstack([ones, s_color, Fbg, Fbr, Fgr, Frr, Fgg, Fbb, Frrr, Frrg, Frrb,
                       Fggg, Fggr, Fggb, Fbbb, Fbbr, Fbbg, Frgb])  # добавил одну степень полинома
    Ftr = F.transpose()  # transpone F
    A = np.matmul(Ftr, F)
    A = np.linalg.inv(A)
    A = np.matmul(A, Ftr)  # get (Ft*F)^-1*Ft

    Ar = np.matmul(A, dr)  # find Ar
    Ag = np.matmul(A, dg)  # find Ag
    Ab = np.matmul(A, db)  # find Ab

    devR = np.matmul(F, Ar)          # get deviation Red
    devG = np.matmul(F, Ag)          # get deviation Green
    devB = np.matmul(F, Ab)          # get deviation Blue
    del_colors = np.vstack([devB.copy(), devG.copy(), devR.copy()]).transpose()     # create mass of deviation
    output_colors = get_result_colors(dist_colors, del_colors)
    return output_colors


# получаем значения скорректированной картинки
# out = ndarray[24,3]
def get_result_colors(dist_color, del_color):
    length = len(dist_color)
    output = np.empty((length,3), dtype='uint8')
    for i in range(length):
        for j in range(3):
            a = dist_color[i,j]-del_color[i,j]
            if a > 255:
                output[i,j] = 255
            elif a < 0.0: output[i,j] = 0
            else: output[i,j] = (dist_color[i,j]-del_color[i,j])//1
    return output

    '''К А Ч Е С Т В О'''
# вычисление евклидового расстояния между 2мя цветами
def evklid_l2_norma(first_color, second_color):
    D = 0
    length = len(first_color)
    for i in range(length):
        a_b = 0
        for j in range(3):
            a = int(first_color[i][j])
            b = int(second_color[i][j])
            a_b1 = pow((a - b), 2)
            a_b += a_b1
        D1 = pow((a_b), 0.5)
        D += D1
    kachestvo_Evklid = D / length
    return(kachestvo_Evklid)

# неправильная вроде, но оценка качества
# из одного (исходного) вычитаю другой (восст) и по 24м цветам беру МО и Дисп
def my_analise_result_noise(first_color, second_color):
    delta = first_color - second_color  # вычисление разницы между исходными и результирующими цветами
    delta_list = np.empty((24, 3), dtype='uint16')
    for i in range(len(delta)):
        for j in range(len(delta[i])):
            if delta[i][j] > 200:
                delta_list[i][j] = 255 - delta[i][j]
            else:
                delta_list[i][j] = delta[i][j]
    # вывод МО и дисперсии от разницы
    print(m + 1, delta_list.mean(), delta_list.var())


def bgr_to_lab_my(in_colors):
    out_colors = []
    for i in range(len(in_colors)):
        sR = int(in_colors[i][2])
        sG = int(in_colors[i][1])
        sB = int(in_colors[i][0])
        Lab = rgb2lab([sR, sG, sB])
        out_colors.append(Lab)
    return out_colors

def to_LAB_cv2(in_colors):
    num_colors = len(in_colors)
    width = 30
    length = 500
    picture = np.empty((num_colors*width, length, 3), dtype='uint8')  # create empty mass for CC
    num_color = 0  # count
    for i in range(0,num_colors*width,width):  # all vertical size, step
        picture[i:i + width,0:length, 0] = in_colors[num_color][0]  # set b channel
        picture[ i:i + width,0:length, 1] = in_colors[num_color][1]  # set g channel
        picture[ i:i + width, 0:length,2] = in_colors[num_color][2]  # set r channel
        num_color += 1

    lab_picture = cv2.cvtColor(picture, cv2.COLOR_BGR2LAB)
    cv2.imshow('BGR', picture)  # show result
    cv2.imshow('LAB', lab_picture)  # show result
    cv2.waitKey(0)

    output = np.empty((num_colors, 3), dtype='uint8')
    for i in range(num_colors):
        output[i] = lab_picture[1+width*i, 0]
    print(output)

CC1 = cv2.imread('CC1.jpg')  # with values
CC2 = cv2.imread('CC2.jpg')  # without values

print('Введи количество цветов для опыта')
n = 24*10
start_colors = np.random.randint(0, 255, (n,3)) # гененируем цвета
#lab_color = to_LAB_cv2(start_colors)
noise3 = np.random.normal(0, 20, 3)  # main, SKO, numbers (B,G,R)
print(noise3)
noise = np.full((len(start_colors),3), noise3)      # double noise3 for 24 colors
#noise = np.random.normal(0, 20, (24,3))  # main, SKO, numbers (24 * B,G,R)

dist_colors = distortion_np(start_colors, noise)     # distortion colors
start_colors_lab = bgr_to_lab_my(start_colors)

for m in range(6):
    # получаю
    res_colors = restruction(start_colors, dist_colors, m)
    res_colors_lab = bgr_to_lab_my(res_colors)
    a = colour.difference.delta_E_CIE2000(start_colors_lab, res_colors_lab)
    CIE_res = 0
    for i in range(len(a)):
        CIE_res += a[i]
    CIE_res = CIE_res/len(start_colors_lab)
    for i in range(len(start_colors)//24):
        generate_result(start_colors[i:i+24], dist_colors[i:i+24], res_colors[i:i+24])

    # вычислим евклидову норму (норму l2))
    print("Качество по евклиду: ",evklid_l2_norma(start_colors,res_colors))
    print("Качество по CIEDE-2000: ",CIE_res, '\n')