# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 11:43:01 2022

@author: ctesc
"""
##### VARIABLES A MODIFIER ######
reset = False
nb_hidden = 8
epochs = 4000

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

os.chdir('C:/Users/ctesc/OneDrive/Desktop/ml_alexis')
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

def baseline_model(nb_hidden=nb_hidden):
    model=Sequential()
    model.add(Dense(nb_hidden, input_shape=(X.shape[1],), activation = 'relu'))
    #model.add(Dense(8, activation = 'relu'))
    model.add(Dense(24, activation = 'softmax'))
    
    myOpt = tf.keras.optimizers.Adam(learning_rate=0.0005, name='Adam')
    model.compile(loss='categorical_crossentropy', optimizer = myOpt, 
                  metrics = ['accuracy'])
    global train_loss
    global val_loss
    global train_acc
    global val_acc
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    
    return model

def train(epochs=epochs, reset=False, nb_hidden=nb_hidden):
    if reset:
        model = baseline_model(nb_hidden)
    else:
        model = load_model(f'model_{nb_hidden}_save')
        
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=64, 
                        validation_data=(X_test, y_test), shuffle=True, verbose=1)
    model.save(f'model_{nb_hidden}_save')
    return history

def update_metrics():
    history = train(epochs=epochs, reset = reset)
    train_loss.append(history.history['loss'])
    val_loss.append(history.history['val_loss'])
    train_acc.append(history.history['accuracy'])
    val_acc.append(history.history['val_accuracy'])   
    train_loss = np.array(train_loss)
    val_loss = np.array(val_loss)
    train_acc = np.array(train_acc)
    val_acc = np.array(val_acc)
    train_loss = train_loss.reshape((train_loss.shape[1], 1))
    val_loss = val_loss.reshape((val_loss.shape[1], 1))
    train_acc = train_acc.reshape((train_acc.shape[1], 1))
    val_acc = val_acc.reshape((val_acc.shape[1], 1))
    metrics = pd.DataFrame(train_loss, columns=['train_loss'])
    metrics['val_loss'] = val_loss
    metrics['train_acc'] = train_acc
    metrics['val_acc'] = val_acc

def graphs(nb_hidden=nb_hidden, metrics=update_metrics()):
    plt.figure(figsize=(12,8))
    plt.plot(metrics['val_acc'], label='val acc')
    plt.plot(metrics['train_acc'], label = 'train acc')
    plt.legend()
    plt.savefig(f'accuracy_{nb_hidden}.png', dpi=200)
    
    plt.figure(figsize=(12,8))
    plt.plot(metrics['val_loss'], label='val loss')
    plt.plot(metrics['train_loss'], label = 'train loss')
    plt.legend()
    plt.savefig(f'loss_{nb_hidden}.png', dpi=200)
    
    return 
