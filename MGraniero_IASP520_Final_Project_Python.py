import pandas as pd
import numpy as np
from scipy import spatial
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns

pd.set_option("display.max_rows", None, "display.max_columns", None) # set max number of rows/columns, when dataframe called, to be displayed as unlimited

data = pd.read_csv('C:\\Users\\Matteo\\Desktop\\IASP 520\\FINAL PROJECT\\column_2C.csv') # read the csv file

df = pd.DataFrame(data=data.values, columns= ['pelvic_incidence', 'pelvic_tilt', 'lumbar_lordosis_angle', 'sacral_slope', 'pelvic_radius', 'degree_spondylolisthesis', 'class'])

df_values = df.iloc[:,:6] # dataframe excluding the feature 'class'

# Here we will extract the labels of class feature into a list to be used later in confusion matrices

y_temp = pd.get_dummies(df['class'])
output_labels = list(y_temp.columns)

labels = output_labels

# df.iloc[:,0:1] shows all instances for pelvic incidence
# df.iloc[:,1:2] shows all instances for pelvic tilt
# df.iloc[:,2:3] shows all instances for lumbar lordosis angle
# df.iloc[:,3:4] shows all instances for sacral slope
# df.iloc[:,4:5] shows all instances for pelvic radius
# df.iloc[:,5:6] shows all instances for degree spondylolisthesis

'--------------------------------- compute similarities between data values ----------------------------------------------'
   
def PI_distances1(): # compares distances between of sets of data found in abnormal class vs normal class
    
    a = np.array(df.iloc[50:100,0:1])
    b = np.array(df.iloc[260:,0:1])
    
    return print('Distance between one set found in abnormal class and \nanother found in normal class for Pelvic Incidence:',np.linalg.norm(a - b))

def PI_distances2(): # compares distances between of sets of data found in normal class vs normal class
    
    a = np.array(df.iloc[210:260,0:1])
    b = np.array(df.iloc[260:,0:1])
    
    return print('Distance between two sets found in \nnormal class for Pelvic Incidence:',np.linalg.norm(a - b))

def PT_distances1(): # compares distances between of sets of data found in abnormal class vs normal class
    
    a = np.array(df.iloc[50:100,1:2])
    b = np.array(df.iloc[260:,1:2])
    
    return print('Distance between one set found in abnormal class and \nanother found in normal class for Pelvic Tilt:',np.linalg.norm(a - b))

def PT_distances2(): # compares distances between of sets of data found in normal class vs normal class
    
    a = np.array(df.iloc[210:260,1:2])
    b = np.array(df.iloc[260:,1:2])
    
    return print('Distance between two sets found in \nnormal class for Pelvic Tilt:',np.linalg.norm(a - b))

def LLA_distances1(): # compares distances between of sets of data found in abnormal class vs normal class
    
    a = np.array(df.iloc[50:100,2:3])
    b = np.array(df.iloc[260:,2:3])
    
    return print('Distance between one set found in abnormal class and \nanother found in normal class for Lumbar Lordosis Angle:',np.linalg.norm(a - b))

def LLA_distances2(): # compares distances between of sets of data found in normal class vs normal class
    
    a = np.array(df.iloc[210:260,2:3])
    b = np.array(df.iloc[260:,2:3])
    
    return print('Distance between two sets found in \nnormal class for Lumbar Lordosis Angle:',np.linalg.norm(a - b))

def DS_distances1(): # compares distances between of sets of data found in abnormal class vs normal class
    
    a = np.array(df.iloc[50:100,5:6])
    b = np.array(df.iloc[260:,5:6])
    
    return print('Distance between one set found in abnormal class and \nanother found in normal class for Lumbar Degree Spondylolisthesis:',np.linalg.norm(a - b))

def DS_distances2(): # compares distances between of sets of data found in normal class vs normal class
    
    a = np.array(df.iloc[210:260,5:6])
    b = np.array(df.iloc[260:,5:6])
    
    return print('Distance between two sets found in \nnormal class for Degree Spondylolisthesis:',np.linalg.norm(a - b))
    
'--------------------------------------- visualize data clustering -------------------------------------------------------'

def PI_Versus_PT(): # plots the pelvic incidence instances vs degree spondylolisthesis
    
    a = plt.scatter(df['pelvic_incidence'],df['pelvic_tilt'])
    km = KMeans(n_clusters = 2) # sets KMeans number to 2, as in 2 clusters assumed (normal and abnormal)
    km.fit_predict(df[['pelvic_incidence','pelvic_tilt']])
    y_predicted = km.fit_predict(df[['pelvic_incidence','pelvic_tilt']])
    df['cluster'] = y_predicted # adds feature 'cluster' to dataset
    
    df1 = df[df.cluster==1] # 1 = abnormal 
    df2 = df[df.cluster==0] # 0 = normal
    
    plt.scatter(df1['pelvic_incidence'],df1['pelvic_tilt'], color='red',label='Abnormal')
    plt.scatter(df2['pelvic_incidence'],df2['pelvic_tilt'], color='green',label='Normal')
    plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid') # selects centroid of each cluster
    
    plt.xlabel('pelvic_incidence')
    plt.ylabel('pelvic_tilt')
    
    plt.legend()
    plt.show()

def PI_Versus_PR():
    
    a = plt.scatter(df['pelvic_incidence'],df['pelvic_radius'])
    km = KMeans(n_clusters = 2)
    km.fit_predict(df[['pelvic_incidence','pelvic_radius']])
    y_predicted = km.fit_predict(df[['pelvic_incidence','pelvic_radius']])
    df['cluster'] = y_predicted
    
    df1 = df[df.cluster==0]
    df2 = df[df.cluster==1]
    
    plt.scatter(df1['pelvic_incidence'],df1['pelvic_radius'], color='red',label='Abnormal')
    plt.scatter(df2['pelvic_incidence'],df2['pelvic_radius'], color='green',label='Normal')
    plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
    
    plt.xlabel('pelvic_incidence')
    plt.ylabel('pelvic_radius')
    
    plt.legend()
    plt.show()

def LLA_Versus_DS():
    
    a = plt.scatter(df['lumbar_lordosis_angle'],df['degree_spondylolisthesis'])
    km = KMeans(n_clusters = 2)
    km.fit_predict(df[['lumbar_lordosis_angle','degree_spondylolisthesis']])
    y_predicted = km.fit_predict(df[['lumbar_lordosis_angle','degree_spondylolisthesis']])
    df['cluster'] = y_predicted
    
    df1 = df[df.cluster==1]
    df2 = df[df.cluster==0]
    
    plt.scatter(df1['lumbar_lordosis_angle'],df1['degree_spondylolisthesis'], color='red',label='Abnormal')
    plt.scatter(df2['lumbar_lordosis_angle'],df2['degree_spondylolisthesis'], color='green',label='Normal')
    plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
    
    plt.xlabel('lumbar_lordosis_angle')
    plt.ylabel('degree_spondylolisthesis')
    
    plt.ylim(-15,200)
    
    plt.legend()
    plt.show()

def SS_Versus_DS():
    
    a = plt.scatter(df['sacral_slope'],df['degree_spondylolisthesis'])
    km = KMeans(n_clusters = 2)
    km.fit_predict(df[['sacral_slope','degree_spondylolisthesis']])
    y_predicted = km.fit_predict(df[['sacral_slope','degree_spondylolisthesis']])
    df['cluster'] = y_predicted
    
    df1 = df[df.cluster==0]
    df2 = df[df.cluster==1]
    
    plt.scatter(df1['sacral_slope'],df1['degree_spondylolisthesis'], color='red',label='Abnormal')
    plt.scatter(df2['sacral_slope'],df2['degree_spondylolisthesis'], color='green',label='Normal')
    plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
    
    plt.xlabel('sacral_slope')
    plt.ylabel('degree_spondylolisthesis')
    
    plt.xlim(0,90)
    plt.ylim(-15,170)
    
    plt.legend()
    plt.show()

def PR_Versus_DS():
    
    a = plt.scatter(df['pelvic_radius'],df['degree_spondylolisthesis'])
    km = KMeans(n_clusters = 2)
    km.fit_predict(df[['pelvic_radius','degree_spondylolisthesis']])
    y_predicted = km.fit_predict(df[['pelvic_radius','degree_spondylolisthesis']])
    df['cluster'] = y_predicted
    
    df1 = df[df.cluster==1]
    df2 = df[df.cluster==0]
    
    plt.scatter(df1['pelvic_radius'],df1['degree_spondylolisthesis'], color='red',label='Abnormal')
    plt.scatter(df2['pelvic_radius'],df2['degree_spondylolisthesis'], color='green',label='Normal')
    plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
    
    plt.xlabel('pelvic_radius')
    plt.ylabel('degree_spondylolisthesis')
    
    plt.ylim(-15,190)
    
    plt.legend()
    plt.show()

'--------------------------------------- random forest classification -------------------------------------------------------'

def Random_Forest():
    
    df.loc[df['class'] == 'Abnormal', 'number'] = 0 # create feature to represent abnormal class as 0
    df.loc[df['class'] == 'Normal', 'number'] = 1 # create feature to represent normal class as 1

    X = df.iloc[:,:6].values # dataframe excluding the features 'class' and 'number'
    y = df.iloc[:,7].values # data frame containing only 'number' feature

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    clf = RandomForestClassifier(n_estimators = 200)
    clf.fit(X_train,y_train)

    y_pred = clf.predict(X_test)

    print(classification_report(y_test,y_pred))
    print(accuracy_score(y_test, y_pred))

    plt.figure(figsize=(12,12)) # Create confusion matrix heat map to show where algorithm predicted 
    sns.set(font_scale=.8)
    sns.heatmap(metrics.confusion_matrix(y_test, y_pred), annot=True, fmt='d', xticklabels=labels, yticklabels=labels)
    plt.title('Random Forest Confusion Matrix')
    plt.xlabel('Predicted Outcome')
    plt.ylabel('Actual Outcome')
    plt.show()






