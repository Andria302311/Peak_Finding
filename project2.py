### extract function from data
import cv2
import numpy
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

### image data to matrix
img = cv2.imread('testHistogram.png')
#img0=cv2.imread('histogram1.png')

### matrix to true/false , blue points are true ,which needed for function node points
### white pixels are false points, which are not part of function
### thats why we need to differentiate each other and present by true/false

print("Project 2 ")
lst=numpy.all(img<[250,250,250],2).tolist()
#lst0=numpy.all(img0<[250,250,250],2).tolist()
print("transformation image 3d array to 2d array true/false ")
print("")
print(lst)
print(lst[0])


### we need to detect peak
### we need to take local maximum of that function ,all local maximum
### local max- peak of function
# true entity are function , we need local max
# we need peak in each column , contioues sequence , all true and last true is peak
# then check this point by next conditions:
# e.i we need points where [x,y] -> [y+1][x]==false [y][x +/- 1]=false ;
### detect all column peak and save in list , that will be function all nodal point

def detectPeakInX(coord_x,coord_y,matrix):
    y=len(matrix)-1
    while(matrix[y-coord_y][coord_x]==True and y-coord_y>0):
        coord_y+=1

    return coord_x,coord_y-1


def detectNodalPoints(lst):
    maxs=[]
    for i in range(0, len(lst[0])):
        max = detectPeakInX(i, 0, lst)
        if (max[1] > 25):
            maxs.append(max)
    return maxs

maxs=detectNodalPoints(lst)

print("")
print("description of function is in maxs array, by node points , (x,y) coordinaates ")
print("")
print(maxs)

#print("second test:")
#maxs0=detectNodalPoints(lst0)


# finding peak in array
# index of that peak will be x coordinate of local max, peak
def localMaximums(maxs):
    IndexOfPeak = []
    peakMax = []
    for i in range(1, len(maxs) - 1):
        if (maxs[i][1] > maxs[i - 1][1] and maxs[i][1] > maxs[i + 1][1]):
            IndexOfPeak.append(maxs[i][0])
            peakMax.append(maxs[i])
        elif (maxs[i - 1][1] < maxs[i][1] == maxs[i + 1][1]):
            for j in range(i, len(maxs) - 1):
                if (maxs[j][1] > maxs[j + 1][1]):
                    IndexOfPeak.append(maxs[j][0])
                    peakMax.append(maxs[j])
                    break
    print(peakMax)
    peakMax = [*set(peakMax)]

    return  peakMax

peakMax=localMaximums(maxs)
#peakMax0=localMaximums(maxs0)

# we find peaks and save it in peakMax


def drawLocalMaximums(maxs,peakMax,img):
    for i in maxs:
        img[len(img) - i[1]][i[0]] = [0, 255, 0]

    for i in peakMax:
        img[len(img) - i[1]][i[0]] = [0, 0, 255]

        cv2.circle(img, (i[0], len(img) - i[1]), 10, (0, 0, 255))
    cv2.imshow('Test image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


drawLocalMaximums(maxs,peakMax,img)

img1 = Image.fromarray(img, 'RGB')
img1.save('extractfunction.png')
img1.show()

#drawLocalMaximums(maxs0,peakMax0,img0)


#img2 = Image.fromarray(img0, 'RGB')
#img2.save('peak1.png')
#img2.show()

###
x = [i[0] for i in maxs]
y = [i[1] for i in maxs]

""" cubic spline """
n = len(x)
B = np.zeros((1, 4 * (n - 1)))
plt.scatter(x, y)
B[0, 0:2 * (n - 1)] = np.array([[y[i], y[i + 1]] for i in range(len(y)) if i + 1 < len(y)]).flatten().reshape(1, 2 * (
        n - 1)).squeeze()
A = np.zeros((4 * (n - 1), 4 * (n - 1)))
for i in range(n - 1):
    coeff = np.array(
        [x[i] ** 3, x[i] ** 2, x[i] ** 1, x[i] ** 0, x[i + 1] ** 3, x[i + 1] ** 2, x[i + 1] ** 1,
         x[i + 1] ** 0]).reshape((2, 4))
    A[(2 * i): (2 * i + 2), (4 * i):(4 * i + 4)] = coeff
    if i < n - 2:
        firstDerivativeCoeff = np.array(
            [3 * x[i + 1] ** 2, 2 * x[i + 1] ** 1, x[i + 1] ** 0, 0, -3 * x[i + 1] ** 2, -2 * x[i + 1] ** 1,
             -x[i + 1] ** 0, 0]).reshape(1, 8)
        A[2 * (n - 1) + i, (4 * i):(4 * i + 8)] = firstDerivativeCoeff
    if i < n - 2:
        secondDerivativeCoeff = np.array(
            [6 * x[i + 1] ** 1, 2 * x[i + 1] ** 0, 0, 0, -6 * x[i + 1] ** 1, -2 * x[i + 1] ** 0, 0, 0, ]).reshape(1, 8)
        A[2 * (n - 1) + (n - 2) + i, (4 * i):(4 * i + 8)] = secondDerivativeCoeff
A[-2, 0:2] = [6 * x[0], 2]
A[-1, -4:-2] = [6 * x[n - 1], 2]

coeffs = np.linalg.solve(A, B.T)
for i in range(n - 1):
    domain = np.linspace(x[i], x[i + 1], 20)
    cubicCoeff = coeffs[(0 + 4 * i):(4 + 4 * i)]
    f = cubicCoeff[0] * (domain ** 3) + cubicCoeff[1] * (domain ** 2) + cubicCoeff[2] * (domain ** 1) + cubicCoeff[
        3] * (domain ** 0)
    plt.plot(domain, f, color="b")
plt.show()
""" """


# check from peak in exact row is real peak, then check row+1  peak  is row
def checkNeighboursNUll(coord_x,coord_y,matrix):

    y=len(matrix)-1

    if(coord_x==len(matrix[y-coord_y])-1):
       return None
    if(matrix[y-coord_y][coord_x+1]==False and matrix[y-coord_y][coord_x-1]==False):
        return checkNeighboursNUll(coord_x+1,coord_y,matrix)

    elif(matrix[y-coord_y][coord_x+1]==True):
        NEW_COORD=detectPeakInX(coord_x+1,coord_y,matrix)
        return checkNeighboursNUll(NEW_COORD[0],NEW_COORD[1],matrix)
    # elif(matrix[y-coord_y][coord_x-1]!=0):
    #     return checkNeighboursNUll(detectPeakInX(coord_x - 1, coord_x, matrix)[0],
    #                                detectPeakInX(coord_x - 1, coord_x, matrix)[1], matrix)

## call checkNeighbour from starting point , find first derivative=0(x,y) , then you need to call  checkneighbour from
## (x,0) and that finds second derivative until x==last x coordinate





# least square apprx
# we need to define coeff
# a(0) and a(1)
# input is list of tuples(x,y coordinates)
#fun=[(1,1.3),(2,3.5),(3,4.2),(4,5),(5,7),(6,8.8),(7,10.1),(8,12.5),(9,13),(10,15.6)]

fun=maxs
SumOfX=sum([i[0] for i in fun])
print(SumOfX)
SumOfY=sum([i[1] for i in fun])
print(SumOfY)
SumOfXX=sum([i[0]*i[0] for i in fun])
print(SumOfXX)
SumOfXY=round(sum([i[1]*i[0] for i in fun]))
print(SumOfXY)
m=len(fun)
print(m)

#formula of a0= SumOfX*SumOfY-SumOfXY*SumOfY / m(SumOfXX) - (SumOfX)^2
a0=(SumOfXX*SumOfY-SumOfXY*SumOfX) / (m*(SumOfXX) - (SumOfX)*SumOfX)
print(a0)

#formula of a1
a1_num = (m*SumOfXY)-SumOfX*SumOfY
a1_den =   (m*SumOfXX- SumOfX*SumOfX)
a1=a1_num/a1_den
print(a1_num)
print(a1_den)
# P(x) = a1 * x - a0 -linear apprx



# p(x) - second degree least square apprx poly - ax^2+bx+c
# SUM( ( Y(i) - Y ( X(i) ))^2 ) - minimaze

# A - matrix of equation coeff
# B - y1,..yn
# x - a , b , c
# A * x = B

# x = A (pseudo_Inverse) * B = A(T) * (A*A(T))^(-1) * B

A=[]
for i in range(0,len(fun)):
    A.append([ fun[i][0]*fun[i][0] ,fun[i][0] , 1])
print(A)

B=[i[1] for i in fun]
B=np.array(B)
print(B)


A=np.array(A)
A_T=A.transpose()
print(A)
print(A_T)
A_A_T=np.matmul(A,A_T)
inv_AAT=np.linalg.inv(A_A_T)
pesudoA=np.matmul(A_T,inv_AAT)

### solve by pseudoInverse

abc=np.matmul(pesudoA, B)
abc=abc.tolist()
print(abc)

# polynomial coeff is X
x1 = [i[0] for i in fun]
y1 = [i[1] for i in fun]

#plt.scatter(x, y)
## local maximums
X=[i[0] for i in peakMax]
Y=[i[1] for i in peakMax]
##

## peak - select max
def max(lst):
    max=lst[0]
    max2=lst[0]
    for i in lst:
        if(max[1]<i[1]):
            max2=max
            max=i

    return  max,max2
Peak=max(peakMax)
print("max")
print(Peak)


x = np.linspace(0,600,50)
y = a1*x+a0

### least square second degree plynomial
poly =abc[0]*x*x -abc[1]*x -abc[2]

plt.scatter(x1, y1)
plt.scatter(X,Y)
plt.scatter(Peak[1][0],Peak[1][1])
plt.scatter(Peak[0][0],Peak[0][1])
plt.plot(x,y)
plt.plot(x,poly)
plt.grid()
plt.show()






