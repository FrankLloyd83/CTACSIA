# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 11:43:01 2022
@author: ctesc
"""
##### VARIABLES A MODIFIER ######
reset = False # Reset reseau neuronnes
nb_hidden = 8 # Nombre neuronnes couche cachee
epochs = 100  # Nombre de batch d entrainement des poids

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.preprocessing import StandardScaler, LabelEncoder
warnings.filterwarnings('ignore')
full_path='C:/Users/ctesc/OneDrive/Desktop/ml_alexis'
os.chdir('C:/Users/ctesc/OneDrive/Desktop/ml_alexis') #Chemin vers repertoire travail
#os.chdir('C:/Users/AlexisCS/Documents/IA/Projet_CTACRIA') #Chemin vers repertoire travail
#pd.set_option('display.max_row', 100)
pd.set_option('display.max_column', 200) #allow to display 200 columns of the dataset: useful to see all features

#------------------------------------------------------------------------------
# Lecture donnees entree

data=pd.read_csv('DN_Full_Voxels_Spectres.csv', sep = ';', decimal=',')
source = data['Source (MeV)']
voxels = data['Fantome']

#------------------------------------------------------------------------------
# Mise en forme des donnees

encoder   = LabelEncoder()
source_tr = encoder.fit_transform(source)
voxels_tr = encoder.fit_transform(voxels)
source_tr = source_tr.reshape((len(source_tr), 1))
voxels_tr = voxels_tr.reshape((len(source_tr), 1))

X = np.concatenate((
                    #source_tr, 
                    voxels_tr, 
                    np.array(data['Pastille 1']).reshape((len(source_tr), 1)), 
                    np.array(data['Pastille 2']).reshape((len(source_tr), 1)), 
                    np.array(data['Pastille 3']).reshape((len(source_tr), 1)), 
                    np.array(data['Pastille 4']).reshape((len(source_tr), 1)), 
                    np.array(data['Pastille 5']).reshape((len(source_tr), 1)), 
                    np.array(data['Pastille 6']).reshape((len(source_tr), 1)),
                    np.array(data['Moyenne reelle (barns)']).reshape((len(source_tr), 1)),
                    np.array(data['Ecart type reel']).reshape((len(source_tr), 1))), axis=1)

y = np.array(data['Angle (degres)'])
scaler = StandardScaler()
X = scaler.fit_transform(X)

for i in range(0, len(y)): # Decoupage des valeurs de sorties
    if y[i]==360:
        y[i] = 0
    y[i] = y[i]/15
    
dummy_y = np_utils.to_categorical(y)
X_train, X_test, y_train, y_test = train_test_split(X, dummy_y, test_size=.2)

#------------------------------------------------------------------------------
# Fonction model baseline

def baseline_model(nb_hidden=nb_hidden):
    model=Sequential()
    model.add(Dense(nb_hidden, input_shape=(X.shape[1],), activation = 'relu'))
    #model.add(Dense(8, activation = 'relu'))
    model.add(Dense(24, activation = 'softmax'))
    
    myOpt = tf.keras.optimizers.Adam(learning_rate=0.0005, name='Adam')
    model.compile(loss='categorical_crossentropy', optimizer = myOpt, 
                  metrics = ['accuracy'])
    # global train_loss
    # global val_loss
    # global train_acc
    # global val_acc
    # train_loss = []
    # val_loss = []
    # train_acc = []
    # val_acc = []
    
    return model

#------------------------------------------------------------------------------
# Entrainement resultats

def train(reset, epochs=epochs, nb_hidden=nb_hidden):
    filename = f'model_{nb_hidden}/model_{nb_hidden}_save'
    if reset:
        model = baseline_model(nb_hidden)
    else:
        model = load_model(filename)
    
        
    memory = model.fit(X_train, y_train, epochs=epochs, batch_size=64, 
                        validation_data=(X_test, y_test), shuffle=True, verbose=1)
    model.save(filename)
    if reset:
        hist = pd.DataFrame(memory.history)
        with open(filename+'_history.csv', mode='w') as f:
            hist.to_csv(f)
    else :
        hist = pd.DataFrame(memory.history)
        with open(filename+'_history.csv', mode='a') as f:
            hist.to_csv(f, header=False)
    return memory.history

#------------------------------------------------------------------------------
# Analyse des resultats

def update_metrics(reset, epochs=100):
# =============================================================================
#     global train_loss
#     global val_loss
#     global train_acc
#     global val_acc
#     global memory
# =============================================================================
    filename = f'model_{nb_hidden}/model_{nb_hidden}_save'
    memory = train(epochs=epochs, reset = reset)
    #print(np.array(memory.history['loss']))
    # train_loss = [float(i) for i in train_loss]
    # val_loss = [float(i) for i in val_loss]
    # train_acc = [float(i) for i in train_acc]
    # val_acc = [float(i) for i in val_acc]
    # train_loss.append(history.history['loss'])
    # val_loss.append(history.history['val_loss'])
    # train_acc.append(history.history['accuracy'])
    # val_acc.append(history.history['val_accuracy'])   
    if reset:
        train_loss = memory['loss']
        val_loss = memory['val_loss']
        train_acc = memory['accuracy']
        val_acc = memory['val_accuracy']
    else:
        hist = pd.read_csv(filename+'_history.csv')
        train_loss = hist.loss
        val_loss = hist.val_loss
        train_acc = hist.accuracy
        val_acc = hist.val_accuracy
# =============================================================================
#         new_train_loss = np.array(memory['loss'])
#         new_train_loss = new_train_loss.reshape((new_train_loss.shape[0], 1))
#         old_train_loss = np.array(hist.loss)
#         old_train_loss = old_train_loss.reshape((old_train_loss.shape[0], 1))
#         train_loss = np.concatenate((old_train_loss, 
#                                      new_train_loss), axis=0)
#         new_val_loss = np.array(memory['val_loss'])
#         new_val_loss = new_val_loss.reshape((new_val_loss.shape[0], 1))
#         old_val_loss = np.array(hist.val_loss)
#         old_val_loss = old_val_loss.reshape((old_val_loss.shape[0], 1))
#         val_loss = np.concatenate((old_val_loss, 
#                                    new_val_loss), axis=0)
#         new_train_acc = np.array(memory['accuracy'])
#         new_train_acc = new_train_acc.reshape((new_train_acc.shape[0], 1))
#         old_train_acc = np.array(hist.accuracy)
#         old_train_acc = old_train_acc.reshape((old_train_acc.shape[0], 1))
#         train_acc = np.concatenate((old_train_acc, 
#                                      new_train_acc), axis=0)
#         new_val_acc = np.array(memory['val_accuracy'])
#         new_val_acc = new_val_acc.reshape((new_val_acc.shape[0], 1))
#         old_val_acc = np.array(hist.val_accuracy)
#         old_val_acc = old_val_acc.reshape((old_val_acc.shape[0], 1))
#         val_acc = np.concatenate((old_val_acc, 
#                                   new_val_acc), axis=0)
# =============================================================================


    # train_loss = np.array(train_loss)
    # val_loss = np.array(val_loss)
    # train_acc = np.array(train_acc)
    # val_acc = np.array(val_acc)
    # train_loss = train_loss.reshape((train_loss.shape[0], 1))
    # val_loss = val_loss.reshape((val_loss.shape[0], 1))
    # train_acc = train_acc.reshape((train_acc.shape[0], 1))
    # val_acc = val_acc.reshape((val_acc.shape[0], 1))
    frame = {'train_loss': train_loss, 'val_loss':val_loss, 'train_acc':train_acc, 'val_acc':val_acc}
    metrics = pd.DataFrame(frame)
    print(metrics)
    
    return metrics

#------------------------------------------------------------------------------
# Trace des graphes

def graphs(reset, epochs=100, nb_hidden=nb_hidden):
    metrics = update_metrics(reset=reset, epochs=epochs)
    plt.figure(figsize=(12,8))
    plt.plot(metrics['val_acc'], label='val acc')
    plt.plot(metrics['train_acc'], label = 'train acc')
    plt.legend()
    plt.savefig(f'model_{nb_hidden}/accuracy_{nb_hidden}.png', dpi=200)
    
    plt.figure(figsize=(12,8))
    plt.plot(metrics['val_loss'], label='val loss')
    plt.plot(metrics['train_loss'], label = 'train loss')
    plt.legend()
    plt.savefig(f'model_{nb_hidden}/loss_{nb_hidden}.png', dpi=200)
    
    return 