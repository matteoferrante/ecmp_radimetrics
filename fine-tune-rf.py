"""Questo codice è per fare il fine tuning della random forest"""


import pandas as pd
import numpy as np
import seaborn as sns
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
ap.add_argument("-d", "--dataset",default=r"C:\Users\matte\Dropbox\fisica_medica\lavori_ieo\ml\radimetrics_train.csv")

arg=vars(ap.parse_args())




os.makedirs(r"esperimenti\plain_rf",exist_ok=True)
report=open(r"esperimenti\plain_rf\report_tuned_rf", "w")

#load data

print(f"[INFO] Reading data from {arg['dataset']}")
X,y=data_to_model(pd.read_csv(arg["dataset"]))

## PLAIN RANDOM FOREST

report.write("ESPERIMENTO 2. FINE-TUNING RANDOMFOREST REGRESSOR:\n")
report.write("\t\t Dati non riscalati\n\n")


scoring = { 'r2': 'r2',"explained_variance_score":'explained_variance',"max error":'max_error'}
#scoring=make_scorer(explained_variance_score,max_error,mean_absolute_error,r2_score)
regr=RandomForestRegressor()

print(f"[INFO] Running cross validation for fine tuning")



param_grid={'bootstrap':[True],'min_samples_leaf': [1,5],'max_features':['auto', 'sqrt'],'n_estimators':[100,500,1000]}

grid_search=GridSearchCV(estimator=regr,param_grid=param_grid,cv=3,n_jobs=-1,verbose=2,scoring=scoring,refit='r2')

grid_search.fit(X,y)

best_params=grid_search.best_params_
best_regr=grid_search.best_estimator_

print(f"[INFO] Best params: {best_params}\n saving best model..")
report.write(f"Params Grid: {param_grid}\n\n")
report.write(f"best_params {best_params}\n\n")

### SAVE MODEL

filename = r'esperimenti\plain_rf\finetuned_rf.sav'
pickle.dump(regr, open(filename, 'wb'))
print(f"[INFO] Model saved")
