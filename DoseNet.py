"""Questo codice crea una rete per la stima della dose"""
import argparse
import os
import pandas as pd
from sklearn.model_selection import RepeatedKFold, cross_val_score, train_test_split
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

from Preprocess.data_to_model import data_to_model
from matplotlib import pyplot

def buildmodel():
    model= Sequential([
        Dense(17, activation="relu"),
        Dense(100, activation="relu"),
        Dropout(0.3),
        Dense(32,activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    return(model)






ap=argparse.ArgumentParser()
ap.add_argument("-d", "--dataset",default=r"C:\Users\matte\Dropbox\fisica_medica\lavori_ieo\ml\radimetrics_train.csv")

arg=vars(ap.parse_args())




os.makedirs(r"esperimenti\dosenet",exist_ok=True)
report=open(r"esperimenti\dosenet\report_dosenet", "w")

#load data

print(f"[INFO] Reading data from {arg['dataset']}")
X,y=data_to_model(pd.read_csv(arg["dataset"]))

## PLAIN RANDOM FOREST

report.write("ESPERIMENTO 1. DOSENET  REGRESSOR:\n")
report.write("\t\t Dati non riscalati\n\n")




model=buildmodel()
opt=Adadelta()

X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.10)


model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae', 'mape'])

history=model.fit(X_train,y_train,epochs=6)

model.evaluate(X_test,y_test)

print(history.history)
#
pyplot.plot(history.history['mse'])
pyplot.plot(history.history['mae'])
pyplot.plot(history.history['mape'])
pyplot.show()

#
# estimator= KerasRegressor(build_fn=buildmodel, epochs=10, batch_size=10, verbose=0)
# kfold= RepeatedKFold(n_splits=5, n_repeats=100)
# results= cross_val_score(estimator, X.values, y.values, cv=kfold, n_jobs=2)  # 2 cpus
# results.mean()  # Mean MSE