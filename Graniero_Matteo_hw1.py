####### Problem 1 ########

import numpy as np

####### Problem 2 ########

def Problem2():
    
    array = np.array(list(range(10)))

    print(array.size)

Problem2()

####### Problem 3 ########

def Problem3():

    myVector = [5] * 10
    myVector[4] = 10

    print(myVector)
    
Problem3()

####### Problem 4 ########

def Problem4():

    array = np.array(list(range(10, 51)))

    print(array)

Problem4()

####### Problem 5 ########

def Problem5():

    array = np.array(list(range(50, 9, -1)))

    print(array)

Problem5()
    
####### Problem 6 ########

def Problem6():

    array = np.zeros((5,5))

    print(array)

Problem6()

####### Problem 7 ########

def Problem7():

    array = np.diag([100,200,300,400,500])

    print(array)

Problem7()

####### Problem 8 ########

def Problem8():

    array = np.random.random((4,3,2))

    print(array)

Problem8()

####### Problem 9 ########

def Problem9():

    array = np.random.random((4,3,2))

    myArray = (array / 2) + 0.5

    print(myArray)
    
Problem9()

####### Problem 10 #######

def Problem10():
    
    array = np.array((3,4))

    myArray = list(map(lambda x: x * i, array[i][i]))

    print(myArray)

Problem10()

####### Problem 11 #######

def Problem11():

    array1 = np.random.randint((2,3))
    array2 = np.random.randint((2,3))

    arrayAdd = np.add(array1,array2)
    
    arraySub = np.subtract(array1,array2)

    arrayMul = np.multiply(array1,array2)

    arrayDiv = np.divide(array1,array2)

    print(array1,array2,arrayAdd,arraySub,arrayMul,arrayDiv)

Problem11()
