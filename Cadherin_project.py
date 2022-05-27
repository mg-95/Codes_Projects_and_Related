import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')

'--------------------- create dataframes from csv files ------------------------------------'

df_fpkm = pd.read_csv(r'C:\Users\Matteo\Desktop\Capstone Work\Focused Data\condensed_fpkm.csv').sort_values(by = 'donor_id')
df_complete = pd.read_csv(r'C:\Users\Matteo\Desktop\Capstone Work\Focused Data\cadherin_data.csv').sort_values(by = 'age', ascending = False)
df_fpkm_CDH = pd.read_csv(r'C:\Users\Matteo\Desktop\Capstone Work\Focused Data\condensed_fpkm_CDH.csv').sort_values(by = 'donor_id')
df_fpkm_PCDH = pd.read_csv(r'C:\Users\Matteo\Desktop\Capstone Work\Focused Data\condensed_fpkm_PCDH.csv').sort_values(by = 'donor_id')

df_condensed_CDH = df_fpkm_CDH.iloc[:20,:]
df_condensed_PCDH = df_fpkm_PCDH.iloc[:20,:]

'---------------------- perform analyses of dataframes -------------------------------------'

def boxplot_CDH():

    cols = ['CDHR4','CDH18','CDH12','CDH10','CDH9','CDH6','CDH12P2','CDHR2','CDHR3','CDH17','CDH23']

    fig, ax = plt.subplots(3, 3, figsize=(15,13))
    k=0

    lines = []
    
    for i in range(3):
        
        for j in range(3):
            
            sns.boxplot(x=df_condensed_CDH['donor_id'], y=df_condensed_CDH[cols[k]], ax=ax[i][j],
                        palette="bright", hue=df_condensed_CDH['act_demented'], showfliers = False)

            ax[i][j].xaxis.grid(True)
            ax[i][j].set_xlabel(xlabel='Donor')

            labels = [item.get_text() for item in ax[i][j].get_xticklabels()]
            a=0
            b=['A','B','C','D','E','F','G']
            
            for label in labels:
                labels[a] = b[a]
                a+=1
                
            ax[i][j].set_xticklabels(labels)
            ax[i][j].tick_params(axis='y', rotation=0)
            ax[i][j].set_ylabel(ylabel=cols[k], rotation=0, labelpad=30)
            ax[i][j].legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=3)
            
            k+=1
            
    fig.tight_layout(pad=3.0)       
    plt.show()

def boxplot_PCDH():

    cols = ['PCDH11Y','PCDHB11','PCDH7','PCDH10','PCDH18','PCDHA1','PCDHA@','PCDHA2','PCDHA3','PCDHA4','PCDHA5']

    fig, ax = plt.subplots(3, 3, figsize=(15,13))
    k=0

    lines = []
    
    for i in range(3):
        
        for j in range(3):
            
            sns.boxplot(x=df_condensed_PCDH['donor_id'], y=df_condensed_PCDH[cols[k]], ax=ax[i][j],
                        palette="bright", hue=df_condensed_CDH['act_demented'], showfliers = False)

            ax[i][j].xaxis.grid(True)
            ax[i][j].set_xlabel(xlabel='Donor')

            labels = [item.get_text() for item in ax[i][j].get_xticklabels()]
            a=0
            b=['A','B','C','D','E','F','G']
    
            for label in labels:
                labels[a] = b[a]
                a+=1
                
            ax[i][j].set_xticklabels(labels)
            ax[i][j].tick_params(axis='y', rotation=0)
            ax[i][j].set_ylabel(ylabel=cols[k], rotation=0, labelpad=35)
            ax[i][j].legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=3)
            
            k+=1
            
    fig.tight_layout(pad=3.0)       
    plt.show()        

def correlation1_CDH():
    
    classCols = ['Age','Sex','education_years','Diagnosis','CDHR4','CDH18','CDH12',
                 'CDH10','CDH9','CDH6','CDH12P2','CDH12P4','CDHR2','CDHR3','CDH17','CDH23',
                 'CDHR1','CDHR5','CDH24']

    df_complete.loc[df_complete['age'] == 77, 'Age'] = 0
    df_complete.loc[df_complete['age'] == 78, 'Age'] = 1
    df_complete.loc[df_complete['age'] == 79, 'Age'] = 2
    df_complete.loc[df_complete['age'] == 81, 'Age'] = 3
    df_complete.loc[df_complete['age'] == 82, 'Age'] = 4
    df_complete.loc[df_complete['age'] == 83, 'Age'] = 5
    df_complete.loc[df_complete['age'] == 84, 'Age'] = 6
    df_complete.loc[df_complete['age'] == 85, 'Age'] = 7
    df_complete.loc[df_complete['age'] == 86, 'Age'] = 8
    df_complete.loc[df_complete['age'] == 87, 'Age'] = 9
    df_complete.loc[df_complete['age'] == 88, 'Age'] = 10
    df_complete.loc[df_complete['age'] == 89, 'Age'] = 11
    df_complete.loc[df_complete['age'] == '90-94', 'Age'] = 12
    df_complete.loc[df_complete['age'] == '95-99', 'Age'] = 13
    df_complete.loc[df_complete['age'] == '>100', 'Age'] = 14
    df_complete.loc[df_complete['sex'] == 'M', 'Sex'] = 0
    df_complete.loc[df_complete['sex'] == 'F', 'Sex'] = 1
    df_complete.loc[df_complete['dsm_iv_clinical_diagnosis'] == "No Dementia", 'Diagnosis'] = 0
    df_complete.loc[df_complete['dsm_iv_clinical_diagnosis'] == "Alzheimer's Disease Type", 'Diagnosis'] = 1
    df_complete.loc[df_complete['dsm_iv_clinical_diagnosis'] == "Multiple Etiologies", 'Diagnosis'] = 2
    df_complete.loc[df_complete['dsm_iv_clinical_diagnosis'] == "Vascular", 'Diagnosis'] = 3
    df_complete.loc[df_complete['dsm_iv_clinical_diagnosis'] == "Other or Unknown Cause", 'Diagnosis'] = 4
    df_complete.loc[df_complete['dsm_iv_clinical_diagnosis'] == "Other Medical", 'Diagnosis'] = 5
    
    plt.figure(figsize=(200,200))
    
    corr = df_complete[classCols].corr()
    sns.heatmap(abs(corr),lw=1,annot=True,cmap="Blues")
    plt.xticks(rotation=30)
    plt.yticks(rotation=0)
    plt.show()

def correlation2_CDH():
    
    classCols = ['Age','Sex','education_years','Diagnosis','CDH8','CDH11',
                 'CDH5','CDH16','CDH3','CDH1','CDH13','CDH15','CDH2','CDH20',
                 'CDH7','CDH19','GCDH','CDH22','CDH26','CDH4']

    
    df_complete.loc[df_complete['age'] == 77, 'Age'] = 0
    df_complete.loc[df_complete['age'] == 78, 'Age'] = 1
    df_complete.loc[df_complete['age'] == 79, 'Age'] = 2
    df_complete.loc[df_complete['age'] == 81, 'Age'] = 3
    df_complete.loc[df_complete['age'] == 82, 'Age'] = 4
    df_complete.loc[df_complete['age'] == 83, 'Age'] = 5
    df_complete.loc[df_complete['age'] == 84, 'Age'] = 6
    df_complete.loc[df_complete['age'] == 85, 'Age'] = 7
    df_complete.loc[df_complete['age'] == 86, 'Age'] = 8
    df_complete.loc[df_complete['age'] == 87, 'Age'] = 9
    df_complete.loc[df_complete['age'] == 88, 'Age'] = 10
    df_complete.loc[df_complete['age'] == 89, 'Age'] = 11
    df_complete.loc[df_complete['age'] == '90-94', 'Age'] = 12
    df_complete.loc[df_complete['age'] == '95-99', 'Age'] = 13
    df_complete.loc[df_complete['age'] == '>100', 'Age'] = 14
    df_complete.loc[df_complete['sex'] == 'M', 'Sex'] = 0
    df_complete.loc[df_complete['sex'] == 'F', 'Sex'] = 1
    df_complete.loc[df_complete['dsm_iv_clinical_diagnosis'] == "No Dementia", 'Diagnosis'] = 0
    df_complete.loc[df_complete['dsm_iv_clinical_diagnosis'] == "Alzheimer's Disease Type", 'Diagnosis'] = 1
    df_complete.loc[df_complete['dsm_iv_clinical_diagnosis'] == "Multiple Etiologies", 'Diagnosis'] = 2
    df_complete.loc[df_complete['dsm_iv_clinical_diagnosis'] == "Vascular", 'Diagnosis'] = 3
    df_complete.loc[df_complete['dsm_iv_clinical_diagnosis'] == "Other or Unknown Cause", 'Diagnosis'] = 4
    df_complete.loc[df_complete['dsm_iv_clinical_diagnosis'] == "Other Medical", 'Diagnosis'] = 5
    
    plt.figure(figsize=(200,200))
    
    corr = df_complete[classCols].corr()
    sns.heatmap(abs(corr),lw=1,annot=True,cmap="Blues")
    plt.xticks(rotation=30)
    plt.yticks(rotation=0)
    plt.show()

def correlation1_PCDH():
    
    classCols = ['Age','Sex','education_years','Diagnosis','PCDH7','PCDH10','PCDH18',
                 'PCDHA1','PCDHA@','PCDHA2','PCDHA3','PCDHA4','PCDHA5','PCDHA6','PCDHA7',
                 'PCDHA8','PCDHA9','PCDHA10','PCDHA11','PCDHA12','PCDHA13','PCDHAC1']
    
    df_complete.loc[df_complete['age'] == 77, 'Age'] = 0
    df_complete.loc[df_complete['age'] == 78, 'Age'] = 1
    df_complete.loc[df_complete['age'] == 79, 'Age'] = 2
    df_complete.loc[df_complete['age'] == 81, 'Age'] = 3
    df_complete.loc[df_complete['age'] == 82, 'Age'] = 4
    df_complete.loc[df_complete['age'] == 83, 'Age'] = 5
    df_complete.loc[df_complete['age'] == 84, 'Age'] = 6
    df_complete.loc[df_complete['age'] == 85, 'Age'] = 7
    df_complete.loc[df_complete['age'] == 86, 'Age'] = 8
    df_complete.loc[df_complete['age'] == 87, 'Age'] = 9
    df_complete.loc[df_complete['age'] == 88, 'Age'] = 10
    df_complete.loc[df_complete['age'] == 89, 'Age'] = 11
    df_complete.loc[df_complete['age'] == '90-94', 'Age'] = 12
    df_complete.loc[df_complete['age'] == '95-99', 'Age'] = 13
    df_complete.loc[df_complete['age'] == '>100', 'Age'] = 14
    df_complete.loc[df_complete['sex'] == 'M', 'Sex'] = 0
    df_complete.loc[df_complete['sex'] == 'F', 'Sex'] = 1
    df_complete.loc[df_complete['dsm_iv_clinical_diagnosis'] == "No Dementia", 'Diagnosis'] = 0
    df_complete.loc[df_complete['dsm_iv_clinical_diagnosis'] == "Alzheimer's Disease Type", 'Diagnosis'] = 1
    df_complete.loc[df_complete['dsm_iv_clinical_diagnosis'] == "Multiple Etiologies", 'Diagnosis'] = 2
    df_complete.loc[df_complete['dsm_iv_clinical_diagnosis'] == "Vascular", 'Diagnosis'] = 3
    df_complete.loc[df_complete['dsm_iv_clinical_diagnosis'] == "Other or Unknown Cause", 'Diagnosis'] = 4
    df_complete.loc[df_complete['dsm_iv_clinical_diagnosis'] == "Other Medical", 'Diagnosis'] = 5
    
    plt.figure(figsize=(200,200))
    
    corr = df_complete[classCols].corr()
    sns.heatmap(abs(corr),lw=1,annot=True,cmap="Blues")
    plt.xticks(rotation=30)
    plt.yticks(rotation=0)
    plt.show()

def correlation2_PCDH():
    
    classCols = ['Age','Sex','education_years','Diagnosis','PCDHAC2','PCDHACT','PCDHB2',
                 'PCDHB4','PCDHB5','PCDHB7','PCDHB8','PCDHB16','PCDHB9','PCDHB10','PCDHB11',
                 'PCDHB12','PCDHB13','PCDHB14','PCDHB18P','PCDHB19P','PCDHB15','PCDHGA1',
                 'PCDHGA2']

    df_complete.loc[df_complete['age'] == 77, 'Age'] = 0
    df_complete.loc[df_complete['age'] == 78, 'Age'] = 1
    df_complete.loc[df_complete['age'] == 79, 'Age'] = 2
    df_complete.loc[df_complete['age'] == 81, 'Age'] = 3
    df_complete.loc[df_complete['age'] == 82, 'Age'] = 4
    df_complete.loc[df_complete['age'] == 83, 'Age'] = 5
    df_complete.loc[df_complete['age'] == 84, 'Age'] = 6
    df_complete.loc[df_complete['age'] == 85, 'Age'] = 7
    df_complete.loc[df_complete['age'] == 86, 'Age'] = 8
    df_complete.loc[df_complete['age'] == 87, 'Age'] = 9
    df_complete.loc[df_complete['age'] == 88, 'Age'] = 10
    df_complete.loc[df_complete['age'] == 89, 'Age'] = 11
    df_complete.loc[df_complete['age'] == '90-94', 'Age'] = 12
    df_complete.loc[df_complete['age'] == '95-99', 'Age'] = 13
    df_complete.loc[df_complete['age'] == '>100', 'Age'] = 14
    df_complete.loc[df_complete['sex'] == 'M', 'Sex'] = 0
    df_complete.loc[df_complete['sex'] == 'F', 'Sex'] = 1
    df_complete.loc[df_complete['dsm_iv_clinical_diagnosis'] == "No Dementia", 'Diagnosis'] = 0
    df_complete.loc[df_complete['dsm_iv_clinical_diagnosis'] == "Alzheimer's Disease Type", 'Diagnosis'] = 1
    df_complete.loc[df_complete['dsm_iv_clinical_diagnosis'] == "Multiple Etiologies", 'Diagnosis'] = 2
    df_complete.loc[df_complete['dsm_iv_clinical_diagnosis'] == "Vascular", 'Diagnosis'] = 3
    df_complete.loc[df_complete['dsm_iv_clinical_diagnosis'] == "Other or Unknown Cause", 'Diagnosis'] = 4
    df_complete.loc[df_complete['dsm_iv_clinical_diagnosis'] == "Other Medical", 'Diagnosis'] = 5
    
    plt.figure(figsize=(230,230))
    
    corr = df_complete[classCols].corr()
    sns.heatmap(abs(corr),lw=1,annot=True,cmap="Blues")
    plt.xticks(rotation=30)
    plt.yticks(rotation=0)
    plt.show()

def correlation3_PCDH():
    
    classCols = ['Age','Sex','education_years','Diagnosis','PCDHGA3','PCDHGB1','PCDHGA4',
                 'PCDHGB2','PCDHGA5','PCDHGB3','PCDHGA6','PCDHGA7','PCDHGB4','PCDHGA8',
                 'PCDHGB5','PCDHGA9','PCDHGB6','PCDHGA10','PCDHGB7','PCDHGA11','PCDHGB8P']

    df_complete.loc[df_complete['age'] == 77, 'Age'] = 0
    df_complete.loc[df_complete['age'] == 78, 'Age'] = 1
    df_complete.loc[df_complete['age'] == 79, 'Age'] = 2
    df_complete.loc[df_complete['age'] == 81, 'Age'] = 3
    df_complete.loc[df_complete['age'] == 82, 'Age'] = 4
    df_complete.loc[df_complete['age'] == 83, 'Age'] = 5
    df_complete.loc[df_complete['age'] == 84, 'Age'] = 6
    df_complete.loc[df_complete['age'] == 85, 'Age'] = 7
    df_complete.loc[df_complete['age'] == 86, 'Age'] = 8
    df_complete.loc[df_complete['age'] == 87, 'Age'] = 9
    df_complete.loc[df_complete['age'] == 88, 'Age'] = 10
    df_complete.loc[df_complete['age'] == 89, 'Age'] = 11
    df_complete.loc[df_complete['age'] == '90-94', 'Age'] = 12
    df_complete.loc[df_complete['age'] == '95-99', 'Age'] = 13
    df_complete.loc[df_complete['age'] == '>100', 'Age'] = 14
    df_complete.loc[df_complete['sex'] == 'M', 'Sex'] = 0
    df_complete.loc[df_complete['sex'] == 'F', 'Sex'] = 1
    df_complete.loc[df_complete['dsm_iv_clinical_diagnosis'] == "No Dementia", 'Diagnosis'] = 0
    df_complete.loc[df_complete['dsm_iv_clinical_diagnosis'] == "Alzheimer's Disease Type", 'Diagnosis'] = 1
    df_complete.loc[df_complete['dsm_iv_clinical_diagnosis'] == "Multiple Etiologies", 'Diagnosis'] = 2
    df_complete.loc[df_complete['dsm_iv_clinical_diagnosis'] == "Vascular", 'Diagnosis'] = 3
    df_complete.loc[df_complete['dsm_iv_clinical_diagnosis'] == "Other or Unknown Cause", 'Diagnosis'] = 4
    df_complete.loc[df_complete['dsm_iv_clinical_diagnosis'] == "Other Medical", 'Diagnosis'] = 5
    
    plt.figure(figsize=(200,200))
    
    corr = df_complete[classCols].corr()
    sns.heatmap(abs(corr),lw=1,annot=True,cmap="Blues")
    plt.xticks(rotation=30)
    plt.yticks(rotation=0)
    plt.show()

def correlation4_PCDH():
    
    classCols = ['Age','Sex','education_years','Diagnosis','PCDHGA12','PCDHGB9P','PCDHGC3',
                 'PCDHGC4','PCDHGC5','PCDH1','PCDH12','PCDH15','PCDH8','PCDH8P1','PCDH17',
                 'PCDH20','PCDH9','PCDH9-AS2','PCDH9-AS3','PCDH9-AS4','PCDH11X','PCDH19',
                 'PCDH11Y']
    
    df_complete.loc[df_complete['age'] == 77, 'Age'] = 0
    df_complete.loc[df_complete['age'] == 78, 'Age'] = 1
    df_complete.loc[df_complete['age'] == 79, 'Age'] = 2
    df_complete.loc[df_complete['age'] == 81, 'Age'] = 3
    df_complete.loc[df_complete['age'] == 82, 'Age'] = 4
    df_complete.loc[df_complete['age'] == 83, 'Age'] = 5
    df_complete.loc[df_complete['age'] == 84, 'Age'] = 6
    df_complete.loc[df_complete['age'] == 85, 'Age'] = 7
    df_complete.loc[df_complete['age'] == 86, 'Age'] = 8
    df_complete.loc[df_complete['age'] == 87, 'Age'] = 9
    df_complete.loc[df_complete['age'] == 88, 'Age'] = 10
    df_complete.loc[df_complete['age'] == 89, 'Age'] = 11
    df_complete.loc[df_complete['age'] == '90-94', 'Age'] = 12
    df_complete.loc[df_complete['age'] == '95-99', 'Age'] = 13
    df_complete.loc[df_complete['age'] == '>100', 'Age'] = 14
    df_complete.loc[df_complete['sex'] == 'M', 'Sex'] = 0
    df_complete.loc[df_complete['sex'] == 'F', 'Sex'] = 1
    df_complete.loc[df_complete['dsm_iv_clinical_diagnosis'] == "No Dementia", 'Diagnosis'] = 0
    df_complete.loc[df_complete['dsm_iv_clinical_diagnosis'] == "Alzheimer's Disease Type", 'Diagnosis'] = 1
    df_complete.loc[df_complete['dsm_iv_clinical_diagnosis'] == "Multiple Etiologies", 'Diagnosis'] = 2
    df_complete.loc[df_complete['dsm_iv_clinical_diagnosis'] == "Vascular", 'Diagnosis'] = 3
    df_complete.loc[df_complete['dsm_iv_clinical_diagnosis'] == "Other or Unknown Cause", 'Diagnosis'] = 4
    df_complete.loc[df_complete['dsm_iv_clinical_diagnosis'] == "Other Medical", 'Diagnosis'] = 5
    
    plt.figure(figsize=(200,200))
    
    corr = df_complete[classCols].corr()
    sns.heatmap(abs(corr),lw=1,annot=True,cmap="Blues")
    plt.xticks(rotation=30)
    plt.yticks(rotation=0)
    plt.show()

'---------------------- perform regression analysis of dataframes -------------------------------------'

## Restart shell after running previous analyses as they take up a lot of memory in program

def logreg():

    x = df_complete.iloc[:,19:]
    y = df_complete['act_demented']

    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1)

    logreg = LogisticRegression()
    logreg.fit(x_train,y_train)

    y_pred = logreg.predict(x_test)

    print(y_pred)
    print()
    print()
    print('Accuracy Score:\n', logreg.score(x_test,y_test))
    print()
    print()
    print(confusion_matrix(y_test,y_pred))

def logreg_conf_matrix():

    df_complete.loc[df_complete['act_demented'] == "No Dementia", 'Diagnosis'] = 0
    df_complete.loc[df_complete['act_demented'] == "Dementia", 'Diagnosis'] = 1

    x = df_complete.iloc[:,19:]
    y = df_complete['Diagnosis']

    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3)

    logreg = LogisticRegression()
    logreg.fit(x_train,y_train)

    y_pred = logreg.predict(x_test)

    cm = confusion_matrix(y_test,y_pred)
    class_label = ["Dementia","No Dementia"]
    dataCM = pd.DataFrame(cm, index=class_label,columns=class_label)
    
    sns.heatmap(dataCM, annot=True, cmap='Blues',linewidths=2,fmt='d')
    
    print('Accuracy Score of Logistic Regression:', logreg.score(x_test,y_test))
    plt.title("Logistic Regression Confusion Matrix",fontsize=15)
    plt.xlabel('Predicted outcome')
    plt.ylabel('Actual outcome')
    plt.show()

'---------------------- perform decision tree analysis of dataframes -------------------------------------'

def decision_tree():

    df_complete.loc[df_complete['dsm_iv_clinical_diagnosis'] == "No Dementia", 'Diagnosis'] = 0
    df_complete.loc[df_complete['dsm_iv_clinical_diagnosis'] == "Alzheimer's Disease Type", 'Diagnosis'] = 1
    df_complete.loc[df_complete['dsm_iv_clinical_diagnosis'] == "Multiple Etiologies", 'Diagnosis'] = 2
    df_complete.loc[df_complete['dsm_iv_clinical_diagnosis'] == "Vascular", 'Diagnosis'] = 3
    df_complete.loc[df_complete['dsm_iv_clinical_diagnosis'] == "Other or Unknown Cause", 'Diagnosis'] = 4
    df_complete.loc[df_complete['dsm_iv_clinical_diagnosis'] == "Other Medical", 'Diagnosis'] = 5

    x = df_complete.iloc[:,19:]
    y = df_complete['Diagnosis']

    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3)

    dtc = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=0)
    dtc.fit(x_train, y_train)
    y_predictDTC = dtc.predict(x_test)

    cm = confusion_matrix(y_test,y_predictDTC)
    class_label = ["No Dementia","Alzheimer's Disease Type","Multiple Etiologies","Vascular","Other or Unknown Cause","Other Medical"]
    dataCM = pd.DataFrame(cm, index=class_label,columns=class_label)
    
    sns.heatmap(dataCM, annot=True, cmap='Blues',linewidths=2,fmt='d')
    
    print('Accuracy of Decision Tree model:', str(metrics.accuracy_score(y_test, y_predictDTC)))
    plt.title('Decision Tree Confusion Matrix',fontsize=15)
    plt.xticks(rotation=30)
    plt.xlabel('Predicted outcome')
    plt.ylabel('Actual outcome')
    plt.tight_layout(pad=3.0)
    plt.show()
    




