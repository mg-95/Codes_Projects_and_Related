import pandas as pd
import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt

pd.set_option("display.max_rows", None, "display.max_columns", None) # set max number of rows/columns, when dataframe called, to be displayed as unlimited

df = pd.read_excel(r'C:\Users\Matteo\Desktop\Bruses Research\ABA RNASeq 10X_GENOMICS_2020 trimmed_means.xlsx',sheet_name='CDHs trimmed_means 10X_GENOMICS')

# Trimmed means: calculated average of a dataset after removing both small % of smallest and largest values from the set; helps eliminate outlier influence on mean and allows for better basis of comparison

df = df.dropna().drop([df.index[25]]) # NaN values dropped and Actb row deleted from dataset

df = df.rename(columns={'feature':'cadherins'}) # rename column 'feature' to name 'cadherins'
df = df.set_index('cadherins') # sets column 'cadherins' as index of data frame

'------------------------------- bar plot ------------------------------------------'

def barplot1():

    ax = df[['108_Pvalb','229_L6 IT CTX']].plot(kind='bar', title ="Cadherin Expressions within Cells of the Mouse Brain", legend=True, fontsize=12)

    ax.set_xlabel("Cadherins", fontsize=12)
    ax.set_ylabel("Cell Cadherin Expression", fontsize=12)

    plt.show()

'------------------------------- distance ------------------------------------------'

def distance1():
    
    a = np.array(df.iloc[:,0])
    b = np.array(df.iloc[:,1])
    
    return print('Distance between data sets found in \ncells 108_Pvalb and 229_L6 IT CTX:',np.linalg.norm(a - b))

def distance2():
    
    a = np.array(df.iloc[:,9])
    b = np.array(df.iloc[:,1])
    
    return print('Distance between data sets found in \ncells 199_L4/5 IT CTX and 229_L6 IT CTX:',np.linalg.norm(a - b))

def distance3():
    
    a = np.array(df.iloc[:,22])
    b = np.array(df.iloc[:,1])
    
    return print('Distance between data sets found in \ncells 223_L6 IT CTX and 229_L6 IT CTX:',np.linalg.norm(a - b))
