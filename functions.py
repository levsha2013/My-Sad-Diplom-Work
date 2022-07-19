import numpy as np
import cv2

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

# преобразование из RGB в Lab
def bgr_to_lab_my(in_colors):
    out_colors = []
    for i in range(len(in_colors)):
        sR = int(in_colors[i][2])
        sG = int(in_colors[i][1])
        sB = int(in_colors[i][0])
        Lab = rgb2lab([sR, sG, sB])     # само преобразование
        out_colors.append(Lab)
    return out_colors

def distortion_np(s_color, noise):
    output = np.zeros((len(s_color),3))
    for i in range(len(s_color)):     # 24 colors
        for j in range(3):      # 3 channels
            if s_color[i, j] + noise[i, j] > 255: output[i, j] = 255    # if sum more than 255
            elif s_color[i, j] + noise[i, j] < 0: output[i, j] = 0      # if sum less than 0
            else: output[i, j] = s_color[i, j] + noise[i, j]            # other cases
    return output

def get_color_from_CC_np(img):
    output = np.empty((24,3), dtype='uint8')       # create empty mass for colors
    if img.shape[1] == 1200:        # if CC = CC1
        for i in range(4):
            for j in range(6):
                output[i*6+j] = img[200 + 180 * i, 200 + 180 * j]   # take a color
    if img.shape[1] == 1408:                           # if CC  = CC2
        for i in range(4):
            for j in range(6):
                output[i*6+j] = img[150 + 223 * i, 150 + 223 * j]   # take a color
    if img.shape[1] == 1000:
        output = np.zeros((140, 3), dtype='uint8')
        for i in range(10):
            for j in range(14):
                output[i*14+j] = img[70 + 65 * i, 95 + 65 * j]
    return output

def restruction(s_color, d_clolor, polin =0, mode = 0, C = 0):
    '''
    :param s_color: start color checker
    :param d_clolor: distortion color checker
    :param mode: 0 - I polin, 1 - II polin, 2 - root-polin
    :return:
    '''
    F = get_F_polin(d_clolor, polin)

    # Чистый МНК без весов
    if mode == 0:

        Ftr = F.transpose()  # transpone F
        A1 = np.matmul(Ftr, F)
        A2 = np.linalg.inv(A1)
        A = np.matmul(A2, Ftr)  # get (Ft*F)^-1*Ft

        Ar = np.matmul(A, s_color[:,2])  # find Ar
        Ag = np.matmul(A, s_color[:,1])  # find Ag
        Ab = np.matmul(A, s_color[:,0])  # find Ab

    # МНК с весами
    if mode == 1:
        Ftr = F.transpose()  # transpone F
        A = np.matmul(Ftr,C)
        A = np.matmul(A, F)
        #A = np.matmul(A, Ftr)  # get (Ft*F)^-1*Ft
        A = np.linalg.inv(A)
        A = np.matmul(A, Ftr)
        A = np.matmul(A, C)  # get (Ft*F)^-1*Ft
        Ar = np.matmul(A, s_color[:,2])  # find Ar
        Ag = np.matmul(A, s_color[:,1])  # find Ag
        Ab = np.matmul(A, s_color[:,0])  # find Ab
    # Ar Ag Ab те самые матрицы, которые нужны для мнк, это построенная модель
    # далее с их помощью можно уже находить дельты для восстановления


    devR = np.matmul(F, Ar)          # get deviation Red
    devG = np.matmul(F, Ag)          # get deviation Green
    devB = np.matmul(F, Ab)          # get deviation Blue

    del_colors = np.vstack([devB.copy(), devG.copy(), devR.copy()]).transpose()     # create mass of deviation
    #output_colors = del_colors.astype('uint8')

    return Ab, Ag, Ar, del_colors


    '''К А Ч Е С Т В О'''

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
    cv2.waitKey()
    #cv2.destroyWindow('result')

def generate_result2(color_1,color_2, color_3, color_4 = 0, name = 'result'):
    output = np.empty((800, 1400, 3), dtype='uint8')  # create empty mass for CC
    num_color = 0  # count
    for i in range(0, 800, 80):  # all vertical size, step Y
        for j in range(0, 1400, 100):  # all horizontal size, step X
            output[i:i + 48, j:j + 50, 0] = color_1[num_color][0]  # set b channel
            output[i:i + 48, j:j + 50, 1] = color_1[num_color][1]  # set g channel
            output[i:i + 48, j:j + 50, 2] = color_1[num_color][2]  # set r channel

            output[i:i + 48, j + 50:j + 100, 0] = color_2[num_color][0]  # set b channel
            output[i:i + 48, j + 50:j + 100, 1] = color_2[num_color][1]  # set g channel
            output[i:i + 48, j + 50:j + 100, 2] = color_2[num_color][2]  # set r channel

            output[i + 40:i + 80, j:j + 100, 0] = color_3[num_color][0]  # set b channel
            output[i + 40:i + 80, j:j + 100, 1] = color_3[num_color][1]  # set g channel
            output[i + 40:i + 80, j:j + 100, 2] = color_3[num_color][2]  # set r channel

            #output[i + 48:i + 80, j + 50:j + 100, 0] = color_3[num_color][0]  # set b channel
            #output[i + 48:i + 80, j + 50:j + 100, 1] = color_3[num_color][1]  # set g channel
            #output[i + 48:i + 80, j + 50:j + 100, 2] = color_3[num_color][2]  # set r channel

            num_color += 1
    cv2.imshow(name, output)  # show result
    cv2.waitKey()
    #cv2.destroyWindow('result')

# + вычисление евклидового расстояния между 2мя цветами
def evklid_l2_norma(first_color, second_color):
    kachestvo_Evklid = []
    length = len(first_color)
    for i in range(length):
        a_b = 0
        for j in range(3):
            a = int(first_color[i][j]+0.0001)
            b = int(second_color[i][j]+0.001)
            a_b1 = pow((a - b), 2)
            a_b += a_b1
        D1 = pow((a_b), 0.5)
        kachestvo_Evklid.append(D1)
    return(kachestvo_Evklid)

def evklid1_l2_norma(first_color, second_color):
    kachestvo_Evklid = []
    length = len(first_color)
    for i in range(length):
        a_b = 0
        for j in range(3):
            a = int(first_color[i][j] + 0.0000001)
            b = int(second_color[i][j] + 0.000001)
            a_b1 = pow((a - b), 2)
            a_b += a_b1
        D1 = pow((a_b), 0.5)
        kachestvo_Evklid.append(D1)
    # считаем среднее
    Ev_res = sum(kachestvo_Evklid) / length  # оценка качества всей коррекции
    return (Ev_res)

def get_F_polin(color, mode = 0):
    num_colors = len(color)
    ones = np.ones((num_colors, 1))
    if mode == 0: F = np.concatenate([color], axis=1, dtype='float64')
    # полином 1 степени
    if mode == 1:
        Fbg = (color[:, 2] * color[:, 1]).reshape(num_colors, 1)
        Fbr = (color[:, 2] * color[:, 0]).reshape(num_colors, 1)
        Fgr = (color[:, 1] * color[:, 0]).reshape(num_colors, 1)
        Frr = (color[:, 0]).reshape(num_colors, 1)**2
        Fgg = (color[:, 1]).reshape(num_colors, 1) ** 2
        Fbb = (color[:, 2]).reshape(num_colors, 1) ** 2
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
    # полином 4 степени недоделанный

    elif mode == 3:
        # root 1
        Fbg = (color[:, 2] * color[:, 1]).reshape(num_colors, 1) ** 0.5
        Fbr = (color[:, 2] * color[:, 0]).reshape(num_colors, 1) ** 0.5
        Fgr = (color[:, 1] * color[:, 0]).reshape(num_colors, 1) ** 0.5
        F = np.hstack([ color, Fbg, Fbr, Fgr])
    elif mode == 4:
        # root 2
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

def function_last(A__, dist_colors, MODE):
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
    return tr_colors

# + создание весовой матрицы в квадрате
def new_squad(error_vec, num, e = 0.1):
    C = np.ones((num, 1))
    summ = 0
    for i in range(num):
        C[i] = 1 / (error_vec[i] + e)
        summ += pow(C[i], 2)

    C2 = np.ones((num, 1))
    for i in range(num):
        f = C[i] / pow(summ, 0.5)
        C2[i] = pow(f,2)
    C3 = np.zeros((num, num))
    for i in range(num):
        C3[i][i] = C2[i]
    return C3

def restruction_with_wight( start_colors_loc, dist_colors_loc, color_differ_loc, MODE=0):
    j = 0       # для выхода из бесконечного цикла
    Ev_res_last = 0 # для выхода из цикла по вырождению
    All_Ev_res = []
    numbers = len(start_colors_loc)
    while True:
        if j != 0:
            Ev_res_last = Ev_res
        j+=1    # добавим итерацию
        matrix = new_squad(color_differ_loc, numbers) # создаем матрицу оценки на основе различия

        B_loc, G_loc, R_loc,res_colors_loc = restruction(start_colors_loc, dist_colors_loc, MODE, 1, matrix)

        colour_differ = evklid_l2_norma(res_colors_loc, start_colors_loc)  # вектор ошибки по CIEDE
        #Ev_res = evklid_l2_norma(res_colors_loc, start_colors_loc)
        # считаем среднее по CIEDE как качество цветокоррекции этой итерации
        Ev_res = 0
        for i in range(numbers):
            Ev_res += colour_differ[i]
        Ev_res = Ev_res / numbers   # оценка качества всей коррекции
        #print(Ev_res)

        All_Ev_res.append(Ev_res)
        if j == 100 or Ev_res == Ev_res_last:
            return B_loc, G_loc, R_loc, min(All_Ev_res)
        color_differ_loc = colour_differ

