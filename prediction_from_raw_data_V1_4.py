# -*- coding: utf-8 -*-
"""
Created on Wed May 25 20:55:54 2022
@author: ctesc
"""

import numpy as np
import pandas as pd
import os
import keras
from sklearn.preprocessing import LabelEncoder, StandardScaler

pd.set_option('display.max_columns', None)

# Chemins repertoires developpeurs
full_path_charly='C:/Users/ctesc/OneDrive/Desktop/ml_alexis'
full_path_alexis='C:/Users/AlexisCS/Documents/IA/Projet_CTACRIA'

os.chdir(full_path_charly) #Chemin vers repertoire travail Charly
#os.chdir(full_path_alexis) # Chemin vers repertoire travail Alexis

# Lecture du fichier res
data = pd.read_csv('listing_MCNP_Voxels.res', delimiter = '\t')
identity = data.fichier # colonne identifiant echantillon
tally = data.tally # colonne numero detecteur
value = data['mean'] # colonne valeur taux de capture

# Recuperation des valeurs de morphologie, source et angle
morpho = [i.split('_')[1].split('+')[0] for i in identity]
source = [i.split('_')[2].split('+')[0] for i in identity]
angle = [int(i.split('_')[3].split('.')[0]) for i in identity]

# Creation du nouveau dataframe
frame = {'Morpho':morpho, 'Source':source, 'Angle':angle, 'Tally':tally, 'TauxCapture':value}
df = pd.DataFrame(frame)

# Suppression des valeurs numeriques de source et des tally inutiles (10.)
to_delete = []
for row in df.index :
    if (df.Source[row].isnumeric()) or (str(df.Tally[row])[0] != '3') :    
        to_delete.append(row)
df = df.drop(to_delete)

#creation tableau pour exploitation
tally = pd.unique(df.Tally).astype(str) #recupere valeur unique tally
values = [np.array(df.TauxCapture[df.Tally == int(key)]) for key in tally] # creation liste taux de capture par tally
values = np.asarray(values).T
moyenne = values.mean(axis=1)
std = values.std(axis=1)
values_norm = np.empty((432,6))
i = 0
for row in values:
    j = 0
    for value in row:
        values_norm[i, j] = (value-row.min())/(row.max()-row.min())
        j += 1
    i += 1

configuration = {key: value for key, value in zip(tally, values_norm.T)}# creation dictionnaire pour associer tally - valeur taux de capture(432)
df = df[df.Tally == 314].drop(['Tally', 'TauxCapture'], axis=1)#suppression colonnes en trop
for i in configuration.keys() : #ajout de colonnes dans dataframe par tally
    df[i] = configuration[i]

#dffloat = df.select_dtypes('float') #selection des taux de captures
df['Moyenne'   ] = moyenne
df['EcartType'] = std

encoder = LabelEncoder()
morpho = encoder.fit_transform(np.array(df.Morpho)).reshape((len(df.Morpho),1))

X = np.concatenate([morpho,
                    np.array(df['354']).reshape((len(df.Morpho),1)),
                    np.array(df['344']).reshape((len(df.Morpho),1)),
                    np.array(df['334']).reshape((len(df.Morpho),1)),
                    np.array(df['324']).reshape((len(df.Morpho),1)),
                    np.array(df['314']).reshape((len(df.Morpho),1)),
                    np.array(df['364']).reshape((len(df.Morpho),1)),
                    np.array(df.Moyenne).reshape((len(df.Morpho),1)),
                    np.array(df.EcartType).reshape((len(df.Morpho),1))],
                   axis=1) 

# Standardisation des donnees (-moyenne, /ecart-type)
scaler = StandardScaler()
X = scaler.fit_transform(X)

model = keras.models.load_model('model_8/model_8_save')

y_label = np.array(360 - df.Angle)
for i in range(0, len(y_label)): 
    if y_label[i]==360:
        y_label[i] = 0
y_pred = model.predict(X)
