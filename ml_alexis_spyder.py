# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 11:43:01 2022

@author: ctesc
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime
import dateutil
import seaborn as sns
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.metrics import categorical_accuracy
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
warnings.filterwarnings('ignore')

#os.chdir('C:/Users/ctesc/OneDrive/Desktop/ml_alexis')
#pd.set_option('display.max_row', 100)
pd.set_option('display.max_column', 200) #allow to display 200 columns of the dataset: useful to see all features

data=pd.read_csv('DonneesNormalisees_Voxels_spectres.csv', sep = ';')
source = data['Source (MeV)']
voxels = data['Voxels']
encoder = LabelEncoder()
source_tr = encoder.fit_transform(source)
voxels_tr = encoder.fit_transform(voxels)
source_tr = source_tr.reshape((len(source_tr), 1))
voxels_tr = voxels_tr.reshape((len(source_tr), 1))

X = np.concatenate((source_tr, 
                    voxels_tr, 
                    np.array(data['Pastille 1']).reshape((len(source_tr), 1)), 
                    np.array(data['Pastille 2']).reshape((len(source_tr), 1)), 
                    np.array(data['Pastille 3']).reshape((len(source_tr), 1)), 
                    np.array(data['Pastille 4']).reshape((len(source_tr), 1)), 
                    np.array(data['Pastille 5']).reshape((len(source_tr), 1)), 
                    np.array(data['Pastille 6']).reshape((len(source_tr), 1)),
                    np.array(data['Moyenne réelle (barns)']).reshape((len(source_tr), 1)),
                    np.array(data['Ecart type réel']).reshape((len(source_tr), 1))), axis=1)

y = np.array(data['Angle (°)'])
scaler = StandardScaler()
X = scaler.fit_transform(X)
for i in range(0, len(y)):
    if y[i]==360:
        y[i] = 0
    y[i] = y[i]/15
dummy_y = np_utils.to_categorical(y)
X_train, X_test, y_train, y_test = train_test_split(X, dummy_y, test_size=.2)

def baseline_model():
    model=Sequential()
    model.add(Dense(8, input_shape=(X.shape[1],), activation = 'relu'))
    #model.add(Dense(8, activation = 'relu'))
    model.add(Dense(24, activation = 'softmax'))
    
    myOpt = tf.keras.optimizers.Adam(learning_rate=0.0005, name='Adam')
    model.compile(loss='categorical_crossentropy', optimizer = myOpt, 
                  metrics = ['accuracy', 'categorical_accuracy'])
    return model

model = baseline_model()
history = model.fit(X_train, y_train, epochs=4000, batch_size=64, 
validation_data=(X_test, y_test), shuffle=True, verbose=1)
metrics = pd.DataFrame(history.history['loss'], columns=['train_loss'])
metrics['val_loss'] = history.history['val_loss']
metrics['train_acc'] = history.history['accuracy']
metrics['val_acc'] = history.history['val_accuracy']
