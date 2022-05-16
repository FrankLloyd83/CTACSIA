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
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')
full_path_charly='C:/Users/ctesc/OneDrive/Desktop/ml_alexis'
full_path_alexis='C:/Users/AlexisCS/Documents/IA/Projet_CTACRIA'
#os.chdir(full_path_charly) #Chemin vers repertoire travail Charly
os.chdir(full_path_alexis) #Chemin vers repertoire travail Alexis

#------------------------------------------------------------------------------
# Lecture donnees entree

data=pd.read_csv('DN_Full_Voxels_Spectres.csv', sep = ';', decimal=',')
source = data['Source (MeV)']
voxels = data['Fantome']

#------------------------------------------------------------------------------
# Mise en forme des donnees

encoder   = LabelEncoder() # convertit des categories de type str vers int
source_tr = encoder.fit_transform(source) # application sur les sources
voxels_tr = encoder.fit_transform(voxels) # application sur les voxels

# Reshape pour compatibilite entre les arrays : vecteurs colonne
source_tr = source_tr.reshape((len(source_tr), 1)) # dimension 0 = longueur du dataset (nb lignes), dimension 1 = 1 colonne
voxels_tr = voxels_tr.reshape((len(voxels_tr), 1))

# On reshape + compile tous les vecteurs colonne pour les variables d'entree
X = np.concatenate((
                    #source_tr, # prise en compte ou non du type de source
                    voxels_tr, 
                    np.array(data['Pastille 1']).reshape((len(source_tr), 1)), 
                    np.array(data['Pastille 2']).reshape((len(source_tr), 1)), 
                    np.array(data['Pastille 3']).reshape((len(source_tr), 1)), 
                    np.array(data['Pastille 4']).reshape((len(source_tr), 1)), 
                    np.array(data['Pastille 5']).reshape((len(source_tr), 1)), 
                    np.array(data['Pastille 6']).reshape((len(source_tr), 1)),
                    np.array(data['Moyenne reelle (barns)']).reshape((len(source_tr), 1)),
                    np.array(data['Ecart type reel']).reshape((len(source_tr), 1))), axis=1)

# Variable cible (angle source reel)
y = np.array(data['Angle (degres)'])

# Standardisation des donnees (-moyenne, /ecart-type)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Creation des bins de sortie
for i in range(0, len(y)): 
    if y[i]==360:
        y[i] = 0
    y[i] = y[i]/15
    
# Transformation du vecteur contenant les bins de l'angle source reel en matrice binaire de dimension (nb_donnes, nb_bins)
y_cat = np_utils.to_categorical(y)

# Decoupage du jeu de donnees en train et test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=.2)

#------------------------------------------------------------------------------
# Fonction model baseline

def baseline_model(nb_hidden):
    model=Sequential() # Creation d'un modele sequentiel vide
    model.add(Dense(nb_hidden, input_shape=(X.shape[1],), activation = 'relu')) # Ajout d'une couche a la sequence
    #model.add(Dense(8, activation = 'relu')) # Ajout d'une deuxieme couche cachee
    model.add(Dense(24, activation = 'softmax')) # Ajout de la couche de sortie
    
    myOpt = tf.keras.optimizers.Adam(learning_rate=0.0005, name='Adam') # Definition de l'algorithme d'optimisation
    model.compile(loss='categorical_crossentropy', optimizer = myOpt, 
                  metrics = ['accuracy']) # Compilation du modele 'baseline' (ie generation aleatoire des poids initiaux)

    return model

#------------------------------------------------------------------------------
# Entrainement du modele

def train(reset, nb_hidden, epochs):
    filename = f'model_{nb_hidden}/model_{nb_hidden}_save'
    if reset:   # reset = True si on decide de reinitialiser les poids
        model = baseline_model(nb_hidden)
    else:       # reset = False si on veut re-entrainer un modele existant
        model = load_model(filename)
    
    # Entrainement du modele    
    memory = model.fit(X_train, y_train, epochs=epochs, batch_size=64, 
                        validation_data=(X_test, y_test), shuffle=True, verbose=1)
    model.save(filename) # Sauvegarde du modele entraine (poids, hyperparametres...)
    
    # Sauvegarde des valeurs des metriques loss et accuracy
    if reset:   # Si reset = True, on ecrase les resultats existants
        hist = pd.DataFrame(memory.history)
        with open(filename+'_history.csv', mode='w') as f:
            hist.to_csv(f)
    else:       # Si reset = False, on ajoute les nouveaux resultats au fichier existant
        hist = pd.DataFrame(memory.history)
        with open(filename+'_history.csv', mode='a') as f:
            hist.to_csv(f, header=False)
    return

#------------------------------------------------------------------------------
# Analyse des resultats

def update_metrics(reset, nb_hidden, epochs):
    
    filename = f'model_{nb_hidden}/model_{nb_hidden}_save'
    # On appelle la fonction d'entrainement pour ecrire les nouveaux resultats
    train(reset = reset, nb_hidden=nb_hidden, epochs=epochs) 
    
    # Recuperation des metriques dans une variable hist
    hist = pd.read_csv(filename+'_history.csv')
    train_loss = hist.loss
    val_loss = hist.val_loss
    train_acc = hist.accuracy
    val_acc = hist.val_accuracy
    
    # Mise en forme en pandas.DataFrame
    frame = {'train_loss': train_loss, 'val_loss':val_loss, 'train_acc':train_acc, 'val_acc':val_acc}
    metrics = pd.DataFrame(frame)
        
    return metrics

#------------------------------------------------------------------------------
# Trace des graphes

def graphs(reset, epochs, nb_hidden):
    # Appel de la fonction update_metrics
    metrics = update_metrics(reset=reset, epochs=epochs, nb_hidden=nb_hidden)
    
    # Plot des accuracy (train, val)
    plt.figure(figsize=(12,8))
    plt.plot(metrics['val_acc'], label='val acc')
    plt.plot(metrics['train_acc'], label = 'train acc')
    plt.legend()
    plt.savefig(f'model_{nb_hidden}/accuracy_{nb_hidden}.png', dpi=200)
    
    #plot des losses (train, val)
    plt.figure(figsize=(12,8))
    plt.plot(metrics['val_loss'], label='val loss')
    plt.plot(metrics['train_loss'], label = 'train loss')
    plt.legend()
    plt.savefig(f'model_{nb_hidden}/loss_{nb_hidden}.png', dpi=200)
    
    return 

#------------------------------------------------------------------------------
# Creation excel resultat 

def resultats(nb_hidden, y=y_test):
    #Recuperation metrics resutlat
    filename = f'model_{nb_hidden}/model_{nb_hidden}_save'
    model = load_model(filename)
    y_pred = model.predict(X_test)
    label = np.argmax(y, axis=1)*15
    frame = {'label': label ,'prediction': np.argmax(y_pred, axis=1)*15, 'confiance':np.max(y_pred, axis=1)*100}
    metrics = pd.DataFrame(frame)   
    metrics.to_csv(filename+'_metrics.csv')
    
    return 
            
 
         
