"""Questo codice è per testare sui primi dati di phyiso -> adatta le colonne e prova un modello """


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
import pickle

import os

import matplotlib.pyplot as plt
import argparse

from sklearn.metrics import make_scorer
from sklearn.metrics import explained_variance_score,max_error,mean_absolute_error,r2_score


from Preprocess.data_to_model import data_to_model

ap=argparse.ArgumentParser()
ap.add_argument("-d", "--dataset",default=r"C:\Users\matte\Dropbox\fisica_medica\lavori_ieo\ml\Molinette_Physico.csv")

arg=vars(ap.parse_args())

### READ DATA


original_data=pd.read_csv(arg["dataset"],sep=";")

to_drop=["ID","Accession Number","Descrizione dello Studio","Device","Acquisition_Type","Protocollo"]
original_data.drop(to_drop,axis=1,inplace=True)

adapt_names={"Data di Nascita":"DOB","Sesso":"Gender","Peso (kg)":"Weight","Altezza (m)":"Height", "KVP SERIE":"kVp", "mAs eff":"Mean_mAs", "Collimazione totale mm":"Collimation"}

### ADATTARE COLONNA FANTOCCIO PER DISCRIMINARE CTDI_Head e Body

original_data.rename(columns=adapt_names,inplace=True)
print(original_data.columns)
## ADAPT DATA
