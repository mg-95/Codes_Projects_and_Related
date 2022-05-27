""" Complete Code """

" ---------- Neural Network for Genome Data ----------- "

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as tts
import matplotlib.pyplot as plt
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model
from keras.layers import Dropout


genomic_data = pd.read_excel(r"C:\Users\Matteo\Desktop\Capstone Work_v2\Capstone_Genomic_Dataset.xlsx")

# copy dataset to separate objects so original dataset isnt effected

gene_features = genomic_data.copy()

# remove data instance that do not contain labels for their data instances and superfluous columns 

gene_features = gene_features[gene_features['geneSymbol'].notna()]

''' create vector representation of labels. This is to turn the string representations of the variable into easily
workable integers'''

data = gene_features['geneSymbol']
# define example
values = array(data)

# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)

# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

# invert first example
inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])

''' continue isolating dataset of extra information '''

gene_features.pop('geneID')
gene_features.pop('Max Value')
gene_features.pop('geneSymbol')

'---------------- normalize the expression data and transform data between 0 and 1 --------------------------'

gene_expressions = pd.DataFrame(gene_features)

x = gene_expressions.iloc[:,1:-1].apply(lambda x: (x-x.min())/ (x.max() - x.min()), axis=0)
x = gene_expressions.iloc[:,1:-1].apply(lambda x: x / x.max())

gene_features = np.array(x, dtype=np.float32)

X_train, X_test = tts(gene_features, test_size=0.2)

Y_train, Y_test = tts(integer_encoded, test_size=0.2)

data = np.concatenate((X_train, Y_train), axis=1)
df = pd.DataFrame(data)
df.rename(columns={df.columns[-1]:'Genes'}, inplace=True)
df = df.drop_duplicates(subset = ['Genes'])
Y = df.filter(regex='Genes')
df = df.drop('Genes', 1)
X_train = df.to_numpy()
Y_train = Y.to_numpy()

data = np.concatenate((X_test, Y_test), axis=1)
df = pd.DataFrame(data)
df.rename(columns={df.columns[-1]:'Genes'}, inplace=True)
df = df.drop_duplicates(subset = ['Genes'])
Y = df.filter(regex='Genes')
df = df.drop('Genes', 1)
X_test = df.to_numpy()
Y_test = Y.to_numpy()

X_train = X_train[-50:]
X_test = X_test[-50:]
Y_train = Y_train[-50:]
Y_test = Y_test[-50:]

'----------neural network construction------------------'

neuralnetwork = Sequential()
neuralnetwork.add(Dense(units=12, activation='relu', input_shape=(None,31)))
neuralnetwork.add(Dropout(0.5))
neuralnetwork.add(Dense(units=8, activation='relu'))
neuralnetwork.add(Dropout(0.5))
neuralnetwork.add(Dense(units=6, activation='relu'))
neuralnetwork.add(Dropout(0.5))
neuralnetwork.add(Dense(units=1, activation='softmax'))

opt = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)

neuralnetwork.summary()

'---------------training autoencoder----------------'

neuralnetwork.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

gene_model = neuralnetwork.fit(X_train, Y_train, epochs=100,
                                 batch_size=200, validation_data=(X_test, Y_test))

'--------- visualize model and performance -------------'


plt.plot(gene_model.history['loss'], label='train')
plt.plot(gene_model.history['val_loss'], label='test')
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.show()

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

" ---------- Neural Network for Cancer Data ----------- "

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as tts
from numpy import array
from numpy import argmax
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from keras.layers import Dropout


breast_cancer_data = pd.read_csv(r"C:\Users\Matteo\Desktop\Capstone Work_v2\data.csv")

# copy dataset to separate objects so original dataset isnt effected

df = breast_cancer_data.copy()

# convert diagnosis to numbers

df.loc[df["diagnosis"] == "B", "diagnosis"] = 0
df.loc[df["diagnosis"] == "M", "diagnosis"] = 1

# drop unnecessary columns

df = df.drop(columns=['Unnamed: 32'])
df = df.drop(columns=['id'])

# normalize the data

scaler = MinMaxScaler(feature_range=(-1,1))

scaler = MinMaxScaler()

scaler.fit(df)

normalized = scaler.transform(df)

df = scaler.inverse_transform(normalized)

# create data dataset with df values

df = np.asarray(df).astype(np.float32)

X, y = df[:,1:], df[:,:1]

X_train, X_test, y_train, y_test = tts(X, y, test_size=0.33, random_state=1)

'----------neural network construction------------------'

neuralnetwork = Sequential()
neuralnetwork.add(Dense(units=1000, activation='relu', input_shape=(30,)))
neuralnetwork.add(Dropout(0.5))
neuralnetwork.add(Dense(units=900, activation='relu'))
neuralnetwork.add(Dropout(0.5))
neuralnetwork.add(Dense(units=800, activation='relu'))
neuralnetwork.add(Dropout(0.5))
neuralnetwork.add(Dense(units=700, activation='relu'))
neuralnetwork.add(Dropout(0.5))
neuralnetwork.add(Dense(units=600, activation='relu'))
neuralnetwork.add(Dropout(0.5))
neuralnetwork.add(Dense(units=500, activation='relu'))
neuralnetwork.add(Dropout(0.5))
neuralnetwork.add(Dense(units=400, activation='relu'))
neuralnetwork.add(Dropout(0.5))
neuralnetwork.add(Dense(units=300, activation='relu'))
neuralnetwork.add(Dropout(0.5))
neuralnetwork.add(Dense(units=100, activation='relu'))
neuralnetwork.add(Dropout(0.5))
neuralnetwork.add(Dense(units=200, activation='relu'))
neuralnetwork.add(Dropout(0.5))
neuralnetwork.add(Dense(units=100, activation='relu'))
neuralnetwork.add(Dropout(0.5))
neuralnetwork.add(Dense(units=100, activation='relu'))
neuralnetwork.add(Dropout(0.5))
neuralnetwork.add(Dense(units=50, activation='relu'))
neuralnetwork.add(Dropout(0.5))
neuralnetwork.add(Dense(units=25, activation='relu'))
neuralnetwork.add(Dropout(0.5))
neuralnetwork.add(Dense(units=1, activation='softmax'))

opt = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)

neuralnetwork.summary()

'---------------training autoencoder----------------'

neuralnetwork.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

gene_model = neuralnetwork.fit(X_train, y_train, epochs=500,
                        batch_size=50, validation_data=(X_test, y_test))

plt.plot(gene_model.history['loss'], label='train')
plt.plot(gene_model.history['val_loss'], label='test')
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.show()

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

" ---------- Random Forest for Cancer Data ----------- "

import numpy as np
import pandas as pd
import sklearn
from numpy import array
from numpy import argmax
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split as tts
from numpy import mean
from numpy import std
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier

breast_cancer_data = pd.read_csv(r"C:\Users\Matteo\Desktop\Capstone Work_v2\data.csv")

# copy dataset to separate objects so original dataset isnt effected

df = breast_cancer_data.copy()

# convert diagnosis to numbers

df.loc[df["diagnosis"] == "B", "diagnosis"] = 0
df.loc[df["diagnosis"] == "M", "diagnosis"] = 1

# drop unnecessary columns

df = df.drop(columns=['Unnamed: 32'])
df = df.drop(columns=['id'])

# normalize the data

scaler = MinMaxScaler(feature_range=(-1,1))

scaler = MinMaxScaler()

scaler.fit(df)

normalized = scaler.transform(df)

df = scaler.inverse_transform(normalized)

# create data dataset with df values

df = np.asarray(df).astype(np.float32)

X, y = df[:,1:], df[:,:1]

X_train, X_test, y_train, y_test = tts(X, y, test_size=0.33, random_state=1)

X,y = X_train, y_train

model = RandomForestClassifier()

n_scores = cross_val_score(model, X, y, scoring='accuracy', n_jobs=-1, error_score='raise')

print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

" ---------- Visualizations for Cancer Data ----------- "

import pandas as pd
import numpy as np
from datetime import datetime

''' import the data and create dataframe containing singular years
and avg for those years '''

data = pd.read_csv(r'C:\Users\Matteo\Desktop\Capstone Work_v2\computerprices.csv')

a = data['DATE']
b = data['PCU334111334111']

years = [datetime.strptime(date,'%m/%d/%Y').year for date in a]
years = pd.DataFrame(data=years)

all_data = pd.concat([years,b], axis=1)
all_data = all_data.rename(columns={0:'years',
                                    'PCU334111334111':'prices'})

c = 1990

avg = pd.DataFrame()

for yrs in all_data['years']:
        
        avg = avg.append({'prices':(sum(np.array(all_data.loc[all_data['years']==c,
                                                        ['prices']]))/len(np.array(all_data.loc[all_data['years']==c,
                                                                                                ['prices']])))},
                   ignore_index=True)
        
        c += 1

        if c > 2022:
            break
        else:
            continue

single_yrs = pd.DataFrame(years[0].unique())
avg = avg['prices'].astype(float)

avg = pd.concat([single_yrs,avg],axis=1)
avg = avg.rename(columns={0:'years'})

''' create visualization of the computer price for single years vs
avg producer price index for computers that year '''

import matplotlib.pyplot as plt

avg.plot(kind = 'line', x = 'years', y = 'prices')
plt.ylabel('Price')
plt.xlabel('Years')
plt.legend(['Price'],loc='upper right')
plt.show()

''' create polynomial regression of the avg data '''

Y = avg['prices'].values
X = avg['years'].values

coefs = np.polyfit(X, Y, 2)

p = np.poly1d(coefs)
plt.plot(X,Y, 'bo', markersize=2)
plt.plot(X,p(X), 'r-')
plt.ylabel('Price')
plt.xlabel('Years')
plt.legend(['Price','Predicted Price'],loc='upper right')
plt.show()
