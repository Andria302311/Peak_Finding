
a=[[False,False,False,False],[True,False,True,False],[True,True,True,True],[True,True,True,True]]

print(a)
#print(a[3][0])

# detect last element in exact column, (coord_x), in matrix
def detectPeakInX(coord_x,coord_y,matrix):
    y=len(matrix)-1
    #print(y-coord_y)
    #print(coord_x)
    #print(matrix[y-coord_y][coord_x])
    print(" ")
    while(matrix[y-coord_y][coord_x]!=False and y-coord_y>0):
        coord_y+=1


    print(coord_x)
    print(coord_y-1)
    return coord_x,coord_y-1

print(detectPeakInX(0,0,a))

maxs=[]
for i in range(0,len(a[0])):
    maxs.append(detectPeakInX(i,0,a))

print(maxs)

# finding peak in array
# index of that peak will be x coordinate of local max, peak

# finding peak in peakarray
# secondary peak




#initial_ycoord=detectPeakInX(1,0,a)[0]

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

print(checkNeighboursNUll(1,1,a))

## call checkNeighbour from starting point , find first derivative=0(x,y) , then you need to call  checkneighbour from
## (x,0) and that finds second derivative until x==last x coordinate

