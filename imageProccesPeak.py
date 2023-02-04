

## take image to np.array

## take image to 1,0 black and white elements

## compare coordinates by isWhite(isNUll)( takes vectors.norm==255)

##
import cv2
import numpy
import numpy as np

import matplotlib.pyplot as plt

img = cv2.imread('histogram.png')

print(img)
###

lst=numpy.all(img!=[255,255,255],2).tolist()
print(lst)

###

x = np.array([[[255,255,255],[255,0,0],[0,0,0]]])

def isNorm255(v):
    if(v[0]==255 and v[1]==255 and v[2]==255):
        return False
    return True
print(isNorm255([50,50,50]))

# if v1 = 255,255,255 - white pixel , v2 is other type
#def compareVectors(v1,v2):


toBinary = lambda t:1 if(isNorm255(t)) else 0
Binary = np.array([toBinary(xi) for xi in img[:][0]])
print(Binary)
## for one dimension works
import numpy as np
mymatrix = np.array([[[-1,  0,  1],
        [-2,  0,  2],
        [-4,  0,  4]],
[[-1,  0,  1],
        [-2,  0,  2],
        [-4,  0,  4]]
                     ])

andria=np.array( [toBinary(x) for x in [i for i in mymatrix[:][0]]])
print(andria)

#Binary1=toBinary(mymatrix)
#print(Binary1)


