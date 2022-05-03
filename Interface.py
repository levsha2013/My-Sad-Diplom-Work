import re
import numpy as np
import cv2

dr = 20
db = 30
dg = 40
set_r = {"r", 'g', 'b'}
# алгоритм по составлению сочетаний с повторениями
# степень полинома +1
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
print(list_of_set)

# как дальше работать с этими сочетаниями
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
    if mode == 'root':
        pattern = r'(\w)\1+'
        a = re.findall(pattern, i)
        a = re.sub(pattern, r'\1', i)
        c = 1
        for j in range(len(i)):
            if i[j] == 'r': c = c * dr
            if i[j] == 'b': c = c * db
            if i[j] == 'g': c = c * dg
        c = pow(c, 1/len(i))
        if c not in result:result.append(c)
print(result, type(result))
a = np.array(result[:], dtype='int32')

print(a, type(a), a.shape)
print(a[0], type(a[0]))





# RGB to Lab
"""def rgb2lab ( inputColor ) :

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


sR = 0
sG = 0
sB = 255
var_R = ( sR / 255 )
var_G = ( sG / 255 )
var_B = ( sB / 255 )

if ( var_R > 0.04045 ):     var_R = pow(( ( var_R + 0.055 ) / 1.055 ), 2.4)
else:                       var_R = var_R / 12.92
if ( var_G > 0.04045 ):     var_G = pow((( var_G + 0.055) / 1.055 ), 2.4)
else:                       var_G = var_G / 12.92
if ( var_B > 0.04045 ):     var_B = pow(( ( var_B + 0.055 ) / 1.055 ), 2.4)
else:                       var_B = var_B / 12.92

var_R = var_R * 100
var_G = var_G * 100
var_B = var_B * 100

X = var_R * 0.4124 + var_G * 0.3576 + var_B * 0.1805
Y = var_R * 0.2126 + var_G * 0.7152 + var_B * 0.0722
Z = var_R * 0.0193 + var_G * 0.1192 + var_B * 0.9505

print(X, Y, Z)


var_X = X #/ Reference_X
var_Y = Y #/ Reference_Y
var_Z = Z #/ Reference_Z

if ( var_X > 0.008856 ): var_X = pow(var_X, ( 1/3 ))
else                   : var_X = ( 7.787 * var_X ) + ( 16 / 116 )
if ( var_Y > 0.008856 ): var_Y =  pow(var_Y, ( 1/3 ))
else                   : var_Y = ( 7.787 * var_Y ) + ( 16 / 116 )
if ( var_Z > 0.008856 ): var_Z =  pow(var_Z, ( 1/3 ))
else                   : var_Z = ( 7.787 * var_Z ) + ( 16 / 116 )

CIE_L = ( 116 * var_Y ) - 16
CIE_a = 500 * ( var_X - var_Y )
CIE_b = 200 * ( var_Y - var_Z )

print(CIE_L, CIE_a, CIE_b)

print(rgb2lab([sR,sG, sB]))"""