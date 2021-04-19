"""Spiegazione:

carico i dati
rimuovo alcune colonne inutili (indici), calcolo l'età e rimuovo la DOB
rimuovo altre colonne (Description, Acquisition Type ecc)
rimuovo i valori Nan in colonne specifiche
Inserisco i valori 0 nei CTDI e DLP quando sono zero


"""


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import cross_val_score

import pickle

from datetime import date
import datetime
from dateutil.relativedelta import relativedelta
import math

import matplotlib.pyplot as plt
import argparse


from Preprocess.RadimetricsPreprocessor import RadimetricsPreprocessor


ap=argparse.ArgumentParser()
ap.add_argument("-d", "--dataset",default=r"C:\Users\matte\Dropbox\fisica_medica\lavori_ieo\ml\radimetrics_train.csv")

arg=vars(ap.parse_args())

report=open("report","w")

#load data

print(f"[INFO] Reading data from {arg['dataset']}")
data=pd.read_csv(arg["dataset"])



def prepare_dataset(data):
    pre=RadimetricsPreprocessor(data)


    pre.basic_filter()   # basic filter -> rimuove indici e calcola l'età

    #definisco colonne da non usare

    to_drop=['Device',"Acquisition_Type","Acquisition_Protocol_Name","Series_Description","mAs_Modulated","CTDIvol_Head_Max_mGy","CTDIvol_Body_Max_mGy"]
    pre.drop_columns(to_drop)

    extra_drop=["Description"] #provo a eliminare anche la description
    pre.drop_columns(extra_drop)



    #nan removing
    subset=['Age', 'Weight','Height','Gender','Filter_Type', "ICRP_103_mSv"]

    pre.dropna(subset=subset)
    data=pre.data

    #per le colonne del CTDI e del DLP sostituisco i nan con zero

    fill_subset=['CTDIvol_Body_mGy', 'CTDIvol_Head_mGy', 'DLP_Body_mGy_cm',
           'DLP_Head_mGy_cm',]

    #per il pitch sostituisco con 1.
    fill_one_subset=["Pitch"]
    pre.fillna(subset=fill_subset,inpute=0.)
    pre.fillna(subset=fill_one_subset,inpute=1.)
    return data


data=prepare_dataset(data)

### ENCODING




report.write(f"{data.columns}")
