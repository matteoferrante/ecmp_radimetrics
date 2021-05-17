"""L'idea di questo script è quella di usare autoKeras per esplorare lo spazio dei parametri dell'architettura"""
import argparse

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping
from sklearn.datasets import fetch_california_housing

import autokeras as ak
from sklearn.model_selection import train_test_split

from Preprocess.data_to_model import data_to_model
import os

ap=argparse.ArgumentParser()
ap.add_argument("-d", "--dataset",default=r"C:\Users\matte\Dropbox\fisica_medica\lavori_ieo\ml\radimetrics_train.csv")

arg=vars(ap.parse_args())




os.makedirs(r"esperimenti\dosenet",exist_ok=True)
report=open(r"esperimenti\dosenet\report_auto_dosenet", "w")

#load data

print(f"[INFO] Reading data from {arg['dataset']}")
X,y=data_to_model(pd.read_csv(arg["dataset"]))



#X=mm.fit_transform(X)


report.write("ESPERIMENTO 1. AUTO DOSENET DOSENET  REGRESSOR:\n")
report.write("\t\t Dati non normalizzati\n\n")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15)

early_stopping = EarlyStopping(patience=10,restore_best_weights=True)


# It tries 10 different models.
reg = ak.StructuredDataRegressor(max_trials=20,loss="mean_absolute_percentage_error",metrics=['mse', 'mae', 'mape'])
# Feed the structured data regressor with training data.
reg.fit(X_train, y_train,epochs=30,validation_split=0.10,batch_size=512)
# Predict with the best model.
predicted_y = reg.predict(X_test)
# Evaluate the best model with testing data.
print(reg.evaluate(X_test, y_test))



model = reg.export_model()
print(model.summary())


print(type(model))  # <class 'tensorflow.python.keras.engine.training.Model'>

try:
    model.save(r"esperimenti\dosenet\auto_dosenet_model.h5")

except Exception:
    model.save(r"esperimenti\dosenet\auto_dosenet_model.h5", save_format="tf")



report.write("\t\t model.summary()\n\n")