import numpy as np ## problem 1

def problem2():
    
    my_vector = np.random.rand(100)

    print("First vector:",my_vector)

def problem3():
    
    my_vector = np.random.rand(100)

    reversed_vector = my_vector[::-1]

    print("First vector:",my_vector,"\nReversed vector:",reversed_vector)

def problem4():

    my_vector = np.random.rand(100)

    reversed_vector = my_vector[::-1]

    reshaped_vector = reversed_vector.reshape(10,10)

    print("Reshaped vector:",reshaped_vector)
    
def problem5():

    my_vector = np.random.rand(100)

    reversed_vector = my_vector[::-1]

    reshaped_vector = reversed_vector.reshape(10,10)
    
    diag_vector = np.diag(reshaped_vector)

    print("Diagonal vector:",diag_vector)

def problem6():
    
    my_vector = np.random.rand(100)

    reversed_vector = my_vector[::-1]

    reshaped_vector = reversed_vector.reshape(10,10)
    
    diag_vector = np.diag(reshaped_vector)

    print(diag_vector.shape)

def problem7():

    my_vector = np.random.rand(100)

    reversed_vector = my_vector[::-1]

    reshaped_vector = reversed_vector.reshape(10,10)
    
    diag_vector = np.diag(reshaped_vector)

    diag_vector2 = np.array(diag_vector)

    reshaped_vector2 = diag_vector2.reshape(2,5)

    print("Reshaped vector:",reshaped_vector2)

def problem8():
    
    my_vector = np.random.rand(100)

    reversed_vector = my_vector[::-1]

    reshaped_vector = reversed_vector.reshape(10,10)
    
    diag_vector = np.diag(reshaped_vector)

    diag_vector2 = np.array(diag_vector)

    reshaped_vector2 = diag_vector2.reshape(2,5)
                          
    m,n = 2,5

    def row_sum(arr):
    
        s = 0

        for i in range(m):

            for j in range(n):

                s += arr[i][j]

            print("sum of row",i,"=",s)

            s = 0

    def col_sum(arr):

        s = 0

        for i in range(n):

            for j in range(m):

                s += arr[j][i]

            print("sum of column",i,"=",s)

            s = 0

    print(row_sum(reshaped_vector2),col_sum(reshaped_vector2))

def problem9():

    my_vector = np.random.rand(100)

    reversed_vector = my_vector[::-1]

    reshaped_vector = reversed_vector.reshape(10,10)
    
    diag_vector = np.diag(reshaped_vector)

    split_arrays = np.array_split(diag_vector, 5)

    print("Split arrays:",split_arrays)

def problem10():

    my_vector = np.random.rand(100)

    reversed_vector = my_vector[::-1]

    reshaped_vector = reversed_vector.reshape(10,10)
    
    diag_vector = np.diag(reshaped_vector)

    split_arrays = np.array_split(diag_vector, 5)

    stacked_arrays = np.vstack((split_arrays[0],split_arrays[1],split_arrays[2],split_arrays[3],split_arrays[4]))

    print("Stacked arrays:",stacked_arrays)

def problem11():

    my_vector = np.random.rand(100)

    reversed_vector = my_vector[::-1]

    reshaped_vector = reversed_vector.reshape(10,10)
    
    diag_vector = np.diag(reshaped_vector)

    split_arrays = np.array_split(diag_vector, 5)

    stacked_arrays = np.vstack((split_arrays[0],split_arrays[1],split_arrays[2],split_arrays[3],split_arrays[4]))

    greater_array = stacked_arrays[stacked_arrays > 0.5]

    print("Array with values greater than 0.5:",greater_array)

def problem12():

    my_vector = np.random.rand(100)

    reversed_vector = my_vector[::-1]

    reshaped_vector = reversed_vector.reshape(10,10)
    
    diag_vector = np.diag(reshaped_vector)

    split_arrays = np.array_split(diag_vector, 5)

    stacked_arrays = np.vstack((split_arrays[0],split_arrays[1],split_arrays[2],split_arrays[3],split_arrays[4]))

    greater_array = stacked_arrays[stacked_arrays > 0.5]

    transposed_array = np.transpose(greater_array)

    print("Transposed array:",transposed_array)

import matplotlib.pyplot as plt ## problem 13

import pandas as pd ## problem 14

final_data = pd.read_csv('C:\\Users\\Matteo\\Desktop\\CISC 540\\Final Work\\final.csv', header=None, names=['column1']) ## problem 15

def problem16():

    new = final_data["column1"]= final_data["column1"].str.split(",", n = 1, expand = True)

    final_data.insert(0,'Size',new[0])

    final_data.insert(1,'Color',new[1])

    del final_data['column1']

    print(final_data)

def problem17():

    final_data.loc[final_data['Color'] == 'red', 'Number'] = 1

    final_data.loc[final_data['Color'] == 'blue', 'Number'] = 2

    final_data.loc[final_data['Color'] == 'green', 'Number'] = 3

    final_data.loc[final_data['Color'] == 'orange', 'Number'] = 4

    print(final_data)
    
def problem18_and_19():

    subplot1 = final_data.iloc[:50,0:2]

    subplot2 = final_data.iloc[50:,0:2]

    fig = plt.figure()

    ax1 = fig.add_subplot(211)

    ax2 = fig.add_subplot(212)

    subplot1.plot(kind = 'scatter', x = 'Size', y = 'Color', color = 'red', ax = ax1, legend = False) # axis labels encased within plot call

    subplot2.plot(kind = 'scatter',x = 'Size',y = 'Color', color = 'blue', ax = ax2, legend = False) # axis labels encased within plot call

    fig.tight_layout()

    ax1.set_xlim([0,30])

    ax2.set_xlim([0,30])

    fig.autofmt_xdate()
    
    plt.show() # make figure plot full screen to see fully spaced x tick values

def problem20():

    subplot1 = final_data.iloc[:50,0:2]

    subplot2 = final_data.iloc[50:,0:2]

    fig = plt.figure()

    ax1 = fig.add_subplot(211)

    ax2 = fig.add_subplot(212)

    subplot1.plot(kind = 'scatter', x = 'Size', y = 'Color', color = 'red', ax = ax1)

    subplot2.plot(kind = 'scatter',x = 'Size',y = 'Color', color = 'blue', ax = ax2)

    plt.legend(['Dot'], loc ="lower right")

    fig.tight_layout()

    ax1.set_xlim([0,30])

    ax2.set_xlim([0,30])

    fig.autofmt_xdate()

    plt.show() # make figure plot full screen to see fully spaced x tick values
    
