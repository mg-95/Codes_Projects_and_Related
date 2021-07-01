import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz', compression='gzip', header=0, sep=',', quotechar='"')

df

header = df.head()

header

# The header row with the feature names is missing in the dataset. 
# to insert feature names, we extract them from here: http://kdd.ics.uci.edu/databases/kddcup99/task.html

df_names = pd.read_csv('http://kdd.ics.uci.edu/databases/kddcup99/kddcup.names')

df_names

# extract the feature names from this dataframe into a list 'feature_names'

feature_names = df_names['back'].str.replace(r":.*","").tolist()
feature_names.append('outcome')
feature_names

# append this list as a header to the dataset.

df.columns = feature_names
df

# As per this article about this dataset: (https://www.ee.ryerson.ca/~bagheri/papers/cisda.pdf), it is a good idea to check for duplicate records
# check and remove any duplicate rows:

df.drop_duplicates(keep='first', inplace = True)
df

# Check if theres any missing (NaN) values that need to be imputed

df.isnull().values.any()

# we extract the outcome feature as y which is 1D array having encoded categorical values as outcomes

lee = preprocessing.LabelEncoder()
y = lee.fit_transform(df['outcome'])

# Here we will extract the labels of outcome feature into a list to be used later in confusion matrices

y_temp = pd.get_dummies(df['outcome'])
output_labels = list(y_temp.columns)

df.dtypes

#Encode the object type features (protocol_type, service, flag)
df = pd.get_dummies(df.iloc[:, 0:41])
df

# Standardize the features

scaler = StandardScaler()

df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
df

x = df.values

#Data split into test train

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=4)

# KNN Classifier

knn = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
knn.fit(x_train, y_train)
y_predictKNN = knn.predict(x_test)

print('Accuracy of KNN Classifier: ' + str(metrics.accuracy_score(y_test, y_predictKNN)))

# KNN Confusion matrix:

labels = output_labels

plt.figure(figsize=(12,12))
sns.set(font_scale=.8)
sns.heatmap(metrics.confusion_matrix(y_test, y_predictKNN), annot=True, fmt='d', xticklabels=labels, yticklabels=labels)
plt.title('K-Nearest Neighbour confusion matrix')
plt.xlabel('Predicted outcome')
plt.ylabel('Actual outcome')
plt.show()

# Support Vector Machine SVM

svc=SVC()
svc.fit(x_train,y_train)
y_predictSVM =svc.predict(x_test)

print('Accuracy of SVM model: ' + str(metrics.accuracy_score(y_test, y_predictSVM)))

# SVM confusion matrix

plt.figure(figsize=(12,12))
sns.set(font_scale=.8)
sns.heatmap(metrics.confusion_matrix(y_test, y_predictSVM), annot=True, fmt='d', xticklabels=labels, yticklabels=labels)
plt.title('Support Vector Machine confusion matrix')
plt.xlabel('Predicted outcome')
plt.ylabel('Actual outcome')
plt.show()

# Perceptron

ppn = Perceptron(max_iter = 40, eta0=0.1, random_state = 1)
ppn.fit(x_train, y_train)
y_predictPNN = ppn.predict(x_test)

print('Accuracy of Perceptron model: ' + str(metrics.accuracy_score(y_test, y_predictPNN)))

# Perceptron confusion matrix
plt.figure(figsize=(12,12))
sns.set(font_scale=.8)
sns.heatmap(metrics.confusion_matrix(y_test, y_predictPNN), annot=True, fmt='d', xticklabels=labels, yticklabels=labels)
plt.title('Perceptron confusion matrix')
plt.xlabel('Predicted outcome')
plt.ylabel('Actual outcome')
plt.show()

# Decision Tree

dtc = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=0)
dtc.fit(x_train, y_train)
y_predictDTC = dtc.predict(x_test)

print('Accuracy of Decision Tree model: ' + str(metrics.accuracy_score(y_test, y_predictDTC)))

# Decision Tree confusion matrix

plt.figure(figsize=(12,12))
sns.set(font_scale=.8)
sns.heatmap(metrics.confusion_matrix(y_test, y_predictDTC), annot=True, fmt='d', xticklabels=labels, yticklabels=labels)
plt.title('Decision Tree confusion matrix')
plt.xlabel('Predicted outcome')
plt.ylabel('Actual outcome')
plt.show()

# Sci-kit learn's Logistic regression

lr = LogisticRegression(n_jobs=-1, solver='saga', tol = 0.1)
lr.fit(x_train, y_train)
y_predictLR = lr.predict(x_test)

print('Accuracy of Logistic Regression model: ' + str(metrics.accuracy_score(y_test, y_predictLR)))

# Logistic Regression confusion matrix

plt.figure(figsize=(12,12))
sns.set(font_scale=.8)
sns.heatmap(metrics.confusion_matrix(y_test, y_predictLR), annot=True, fmt='d', xticklabels=labels, yticklabels=labels)
plt.title('Logistic Regression confusion matrix')
plt.xlabel('Predicted outcome')
plt.ylabel('Actual outcome')
plt.show()
