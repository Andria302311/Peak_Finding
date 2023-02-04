import numpy
import numpy as np
import project2
import matplotlib.pyplot as plt
import numpy as np



# least square apprx
# we need to define coeff
# a(0) and a(1)
# input is list of tuples(x,y coordinates)
#fun=[(1,1.3),(2,3.5),(3,4.2),(4,5),(5,7),(6,8.8),(7,10.1),(8,12.5),(9,13),(10,15.6)]

fun=project2.maxs
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
X=[i[0] for i in project2.peakMax]
Y=[i[1] for i in project2.peakMax]
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
Peak=max(project2.peakMax)
print("max")
print(Peak)


x = np.linspace(0,600,50)
y = a1*x+a0
poly =abc[0]*x*x -abc[1]*x -abc[2]

plt.scatter(x1, y1)
plt.scatter(X,Y)
plt.scatter(Peak[1][0],Peak[1][1])
plt.scatter(Peak[0][0],Peak[0][1])
plt.plot(x,y)
#plt.plot(x,poly)
plt.grid()
plt.show()




