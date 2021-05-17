""" QUESTO è per testare su un dataset interno allo ieo"""
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

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

from tensorflow import keras

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))*100


ap=argparse.ArgumentParser()
ap.add_argument("-d", "--dataset",default=r"C:\Users\matte\Dropbox\fisica_medica\lavori_ieo\ml\radimetrics_test.csv")

arg=vars(ap.parse_args())

print(f"[INFO] Reading data from {arg['dataset']}")
X,y=data_to_model(pd.read_csv(arg["dataset"]))


print(X.mean())



print(f"[INFO] {X.shape}")



def results(y,y_pred,report):


    ## OTHER PREDICTIONS

    # MEAN ABSOLUTE ERRROR
    mae=mean_absolute_error(y,y_pred)
    print(f"[RESULT] mean_absolute_error score: {mae} mSv")




    # MAX ERROR
    max_er=max_error(y,y_pred)
    print(f"[RESULT] max_error score: {max_er} mSv")

    # MEAN PERCENTAGE ABSOLUTE ERRROR
    mape=mean_absolute_percentage_error(y,y_pred)
    print(f"[RESULT] mean_percentage_error score: {mape} %")


    #BIAS MAPE

    y_bias_true=[]
    y_bias_pred=[]
    for (i,v) in enumerate(y):
        if v>1:
            y_bias_true.append(v)
            y_bias_pred.append(y_pred[i])


    # MEAN PERCENTAGE ABSOLUTE ERRROR BIAS >1
    mape_b=mean_absolute_percentage_error(y_bias_true,y_bias_pred)
    print(f"[RESULT] mean_percentage_error biased score: {mape_b} %")


    if report is not None:
        report.write(f"[RESULT] mean_absolute_error score: {mae} mSv\n")
        report.write(f"[RESULT] max_error score: {max_er} mSv\n")
        report.write(f"[RESULT] mean_percentage_error score: {mape} %\n")
        report.write(f"[RESULT] mean_percentage_error biased score: {mape_b} %")
    return mae,max_er,mape,mape_b


### CARICO TUTTI I MODELLI FACCIO LE PREVISIONI E SPUTO I RISULTATI

# load the model from disk
#per quelli che usano dati riscalati
mm=MinMaxScaler()

X_norm=mm.fit_transform(X)

### RANDOM FOREST


report=open("internal_test_report_nothr", "w")

print(f"[RANDOM FOREST]\n")
filename=r"esperimenti\plain_rf\plain_rf.sav"
loaded_model = pickle.load(open(filename, 'rb'))
y_pred=loaded_model.predict(X)


data=pd.concat([X,y],axis=1)
data["prediction_rf"]=y_pred

result = loaded_model.score(X, y)

print(f"[RESULT] r2 score {result}")

report.write("RANDOM FOREST MODEL\n")
results(y,y_pred,report=report)
report.write("\n\n\n")


#### FINE TUNED RANDOM FOREST



print(f"[FINE TUNED RANDOM FOREST]\n")
filename=r"esperimenti\plain_rf\finetuned_last_rf.sav"
loaded_model = pickle.load(open(filename, 'rb'))
y_pred=loaded_model.predict(X)


data=pd.concat([X,y],axis=1)
data["prediction_rf_finetuned"]=y_pred

result = loaded_model.score(X, y)

print(f"[RESULT] r2 score {result}")

report.write("FINE TUNED RANDOM FOREST MODEL\n")
results(y,y_pred,report=report)
report.write("\n\n\n")




### SVM
print("[SUPPORT VECTOR MACHINE]\n")

filename=r"esperimenti\svm\svm.sav"
loaded_model = pickle.load(open(filename, 'rb'))
y_pred=loaded_model.predict(X_norm)


data=pd.concat([data,y],axis=1)
data["prediction_svm"]=y_pred


## support vector usa dati riscalati


result = loaded_model.score(X_norm, y)

print(f"[RESULT] r2 score {result}")

report.write("\n\n\t\tSUPPORT VECTOR MACHINE MODEL \n")
results(y,y_pred,report=report)
report.write("\n\n\n")





#### DOSENET
print("\n\n[DOSENET]\n")

loaded_model = keras.models.load_model(r"esperimenti\dosenet\dosenet_model.h5")
y_pred=loaded_model.predict(X_norm)


data=pd.concat([data,y],axis=1)
data["prediction_dosenet"]=y_pred

result = loaded_model.evaluate(X_norm, y)

print(f"[RESULT] model evaluation {result}")

report.write("\n\n\t\tDOSENET \n")
results(y,y_pred,report=report)
report.write("\n\n\n")


data.to_csv("test_processed.csv", index=False)

report.close()
### CHECK REPORT E SALVATAGGIO -> RETE OVERFITTING


##### SOLO PER RANDOM FOREST FINE TUNED AGGIUNGO FEATURE IMPORTANCES E SALVO LA SUA IMMAGINE

