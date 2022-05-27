# -*- coding: utf-8 -*-
"""
Created on Wed May 25 20:55:54 2022

@author: ctesc
"""

import numpy as np
import pandas as pd
import os
import string

def is_int(x):
    try:
        int(x)
    except ValueError:
        return True
    return False

full_path_charly='C:/Users/ctesc/OneDrive/Desktop/ml_alexis'

os.chdir(full_path_charly) #Chemin vers repertoire travail Charly

data = pd.read_csv('listing_MCNP_Voxels.res', delimiter = '\t')
identity = data.fichier
tally = data.tally
value = data['mean']

morpho = [i.split('_')[1].split('+')[0] for i in identity]
source = [i.split('_')[2].split('+')[0] for i in identity]
angle = [int(i.split('_')[3].split('.')[0]) for i in identity]

frame = {'Morpho':morpho, 'Source':source, 'Angle':angle, 'Tally':tally, 'TauxCapture':value}
df = pd.DataFrame(frame)
to_delete = []
for row in df.index:
    if df.Source[row].isnumeric():    
        to_delete.append(row)
df = df.drop(to_delete)
