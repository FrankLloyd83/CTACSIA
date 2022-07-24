# Script de transformation de listing en donnees pour analyse statistique des taux de capture


# ------------------------------------------------------------------------------
# Récupération des données de base
# 

import numpy as np

fichier = 'listing_MCNP_Voxels.res'
donnees = {}
donnees_norm = {}
with open(fichier) as entree :
	for ligne in entree :

		if 'fantome_' in ligne :
			l = ligne.split('\t')
			fichier = l[0][:-2] # Contient geometrie fantome - Energie(MeV/spectres)- Angle(°)
			tally = l[3]        # Numero du tally
			valeur = l[6]       # Taux de captures

			fantome, energie, angle = fichier.replace('+','_').split('_')[1::2] # Separation variables
			i = [fantome, energie, angle] 										# Identifiant ligne
			i[2] = 360 - int(i[2])												# Converti angle horaire vers trigonometrique
			j = tuple(i) 										                # Identifiant ligne
			
			if j not in donnees : donnees[ j ] = [None]*6 						# Empeche doublons	
			
			if 'M1' in fichier :
				if   tally == '354' : donnees[ j ][0] = valeur # ajoute tx capt ds dic et modifie numero capteur
				elif tally == '344' : donnees[ j ][1] = valeur
				elif tally == '334' : donnees[ j ][2] = valeur
				elif tally == '324' : donnees[ j ][3] = valeur
				elif tally == '314' : donnees[ j ][4] = valeur
				elif tally == '364' : donnees[ j ][5] = valeur
			else : 
				if   tally == '364' : donnees[ j ][0] = valeur # ajoute tx capt ds dic et modifie numero capteur
				elif tally == '354' : donnees[ j ][1] = valeur
				elif tally == '344' : donnees[ j ][2] = valeur
				elif tally == '334' : donnees[ j ][3] = valeur
				elif tally == '324' : donnees[ j ][4] = valeur
				elif tally == '314' : donnees[ j ][5] = valeur
# for (fantome, energie, angle) in donnees :
#    values = [str((float(i)-float(min(donnees.get((fantome, energie, angle))))))/((float(max(donnees.get((fantome, energie, angle))))-float(min(donnees.get((fantome, energie, angle)j))))) for i in donnees.get((fantome, energie, angle))]
#    donneesNorm[ (fantome, energie, angle) ] = values
   
donnees_norm = np.empty((len(donnees), 8)) 
i = 0
for fantome, energie, angle in donnees :
    valeurs = [float(i) for i in donnees[(fantome, energie, angle)]]
    donnees_norm[i,:-2] = [(float(j)-min(valeurs))/(max(valeurs)-min(valeurs)) for j in valeurs ]
    donnees_norm[i,-2] = np.mean(valeurs)
    donnees_norm[i,-1] = np.std(valeurs)
    i = i+1

dict_donnees_norm = {}
for key,value in zip(donnees.keys(), donnees_norm) :
    value_str = [str(i) for i in value]
    dict_donnees_norm[key] = value_str


# ------------------------------------------------------------------------------
# Création des données exploitables
# 

# ..............................................................................
# Données de base

with open('DonneesNormalisees_Voxels.csv', 'w', encoding='UTF-8') as sortie :
    for fantome, energie, angle in dict_donnees_norm :
        sortie.write('\t'.join( [fantome, energie, str(angle)] + dict_donnees_norm[(fantome, energie, angle)] )+'\n')
