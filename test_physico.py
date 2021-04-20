"""Questo codice è per testare sui primi dati di phyiso -> adatta le colonne e prova un modello """

# mi sta dando problemi perché mi cambia la media dei valori?



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

from Preprocess.RadimetricsPreprocessor import RadimetricsPreprocessor
from Preprocess.data_to_model import data_to_model

ap=argparse.ArgumentParser()
ap.add_argument("-d", "--dataset",default=r"C:\Users\matte\Dropbox\fisica_medica\lavori_ieo\ml\Molinette_Physico.csv")

arg=vars(ap.parse_args())


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))


def adapt_ctdi_dlp(data,phantom_column="Fantoccio"):


    for i in range(len(data)):

        if "body" in data[phantom_column].iloc[i].lower():
            ctdi=data['CTDI vol SERIE mGy'].iloc[i]
            dlp=data['DLP SERIE mGy*cm'].iloc[i]

            data["CTDIvol_Body_mGy"].iloc[i]=ctdi
            data["DLP_Body_mGy_cm"].iloc[i]=dlp

        elif "head" in data[phantom_column].iloc[i].lower():
            ctdi = data['CTDI vol SERIE mGy'].iloc[i]
            dlp = data['DLP SERIE mGy*cm'].iloc[i]

            data["CTDIvol_Head_mGy"].iloc[i]=ctdi
            data["DLP_Head_mGy_cm"].iloc[i]=dlp

        else:
            print("f[INFO] Warning - can't associate phantom")

    return data.drop(phantom_column,axis=1)


def fix_current(data,def_min=70.4,def_max=147,max_ma_column='CORRENTE MASSIMA SERIE mA',time_column='TEMPO DI ROTAZIONE PER SERIE s'):
    for i in range(len(data)):
        # non posso calcolare il minimo

        max_ma=data[max_ma_column].iloc[i]
        t=float(data[time_column].iloc[i].replace(',', '.'))  #0,5 -> 0.5
        max_mas=max_ma*t
        data["Min_mAs"].iloc[i] = def_min
        data['Max_mAs'].iloc[i] = max_mas

    return data.drop([max_ma_column],axis=1)


### READ DATA


original_data=pd.read_csv(arg["dataset"],sep=";")
#per qualche motivo hanno la virgola -> sostituisco


## ADAPT DATA

original_data.dropna(inplace=True,subset=["Accession Number"])

to_drop=["ID","Accession Number","Descrizione dello Studio","Device","Acquisition_Type","Protocollo"]
original_data.drop(to_drop,axis=1,inplace=True)


print(f"[INFO] {len(original_data)}")


### ADATTARE COLONNA FANTOCCIO PER DISCRIMINARE CTDI_Head e Body
original_data=adapt_ctdi_dlp(original_data)
original_data=fix_current(original_data)


## CAMBIO I NOMI DELLE COLONNE
adapt_names={"Data di Nascita":"DOB","Sesso":"Gender","Peso (kg)":"Weight","Altezza (m)":"Height", "KVP SERIE":"kVp", "mAs eff":"Mean_mAs", "Collimazione totale mm":"Collimation", 'CORRENTE MEDIA SERIE mA':'Nominal_mA', 'TEMPO DI ROTAZIONE PER SERIE s':'Rotation_Time', '_Filter Type':'Filter_Type', 'Dose Efficace serie VP(ICRP 103) mSv':"ICRP_103_mSv"}



original_data.rename(columns=adapt_names,inplace=True)

##ELIMINO LE ULTIME COSE INUTILI
final_drop=['Collimazione singolo elemento mm','Series_Description','mAs_Modulated','CTDIvol_Head_Max_mGy','CTDIvol_Body_Max_mGy','CTDI vol SERIE mGy', 'DLP SERIE mGy*cm']

original_data.drop(final_drop,axis=1,inplace=True)




# , in .
original_data=original_data.applymap(lambda x: (str(x)).replace(',','.'))


string_columns=["DOB","Gender","Filter_Type"]

for c in original_data.columns:
    if c not in string_columns:
        original_data[c] = pd.to_numeric(original_data[c],errors="coerce")


## FIX COLUMNS DTYPES




# PREPROCESSING -> AGE
pre=RadimetricsPreprocessor(original_data) #calcolo età


print(original_data.isna().sum())

## DA QUA MODELLO
X,y=data_to_model(original_data)


print(X.mean())

data=pd.concat([X,y],axis=1)



### QUA CARICO FINALMENTE IL MODELLO


# load the model from disk

filename=r"esperimenti\plain_rf\plain_rf.sav"
loaded_model = pickle.load(open(filename, 'rb'))
y_pred=loaded_model.predict(X)

data["prediction"]=y_pred
data.to_csv("physico_processed.csv", index=False)
result = loaded_model.score(X, y)
print("r2:",r2_score(y,y_pred))


# MEAN ABSOLUTE ERRROR
mae=mean_absolute_error(y,y_pred)
print(f"mean_absolute_error score: {mae}")




# MAX ERROR
max_er=max_error(y,y_pred)
print(f"max_error score: {max_er}")

# MEAN ABSOLUTE ERRROR
mape=mean_absolute_percentage_error(y,y_pred)
print(f"mean_percentage_error score: {mape}")
