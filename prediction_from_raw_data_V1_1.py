# -*- coding: utf-8 -*-
"""
Created on Wed May 25 20:55:54 2022

@author: ctesc
"""

import numpy as np
import pandas as pd
import os

# Chemins repertoires developpeurs
full_path_charly='C:/Users/ctesc/OneDrive/Desktop/ml_alexis'
full_path_alexis='C:/Users/AlexisCS/Documents/IA/Projet_CTACRIA'

os.chdir(full_path_charly) #Chemin vers repertoire travail Charly
# os.chdir(full_path_alexis) # Chemin vers repertoire travail Alexis

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

# Suppression des valeurs numeriques de source
to_delete = []
for row in df.index:
    if df.Source[row].isnumeric():    
        to_delete.append(row)
df = df.drop(to_delete)