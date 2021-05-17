"""Questo codice crea una rete per la stima della dose"""
import argparse
import os
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.regularizers import l1
from sklearn.model_selection import RepeatedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

from Preprocess.data_to_model import data_to_model
from matplotlib import pyplot

import numpy as np

def buildmodel():
    model= Sequential([
        Dense(16, activation="sigmoid"),
        Dense(64, activation="elu"),
        Dropout(0.2),
        Dense(256, activation="elu"),
        Dense(512, activation="elu"),
        Dropout(0.3),
        Dense(64,activation="elu"),
        Dense(1)
    ])
    #model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    return(model)






ap=argparse.ArgumentParser()
ap.add_argument("-d", "--dataset",default=r"C:\Users\matte\Dropbox\fisica_medica\lavori_ieo\ml\radimetrics_train.csv")

arg=vars(ap.parse_args())




os.makedirs(r"esperimenti\dosenet",exist_ok=True)
report=open(r"esperimenti\dosenet\report_dosenet", "w")

#load data

print(f"[INFO] Reading data from {arg['dataset']}")
X,y=data_to_model(pd.read_csv(arg["dataset"]))


mm=MinMaxScaler()
X=X.values
#X=mm.fit_transform(X)


report.write("ESPERIMENTO 1. DOSENET  REGRESSOR:\n")
report.write("\t\t Dati non normalizzati\n\n")




model=buildmodel()
#opt=Adadelta()

EPOCHS=250
BS=256
opt=Adam(lr=1e-3)

X_train, X_test, y_train, y_test = train_test_split(X, y.values, test_size=0.30)
#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15)


print(f"[INFO] {X_train.shape}")

model.compile(loss='mean_absolute_percentage_error', optimizer=opt, metrics=['mse', 'mae', 'mape'])


early_stopping = EarlyStopping(patience=50,restore_best_weights=True)
#
history=model.fit(X_train,y_train,epochs=EPOCHS,validation_split=0.10,batch_size=BS,callbacks=[early_stopping])
#
#model.evaluate(X_test,y_test)
#


model.save(r"esperimenti\dosenet\dosenet_model.h5")

print(f"[INFO] Model evalutaion: {model.evaluate(X_test,y_test)}")
y_pred=model.predict(X_test)
print(f"[INFO] Info about predictions: m: {np.mean(y_pred)}\t std: {np.std(y_pred)}\n original m: {np.mean(y_test)}\t std: {np.std(y_test)}")

# #
pyplot.plot(history.history['mse'])
pyplot.plot(history.history['mae'])
pyplot.plot(history.history['mape'])
pyplot.legend()
pyplot.show()


