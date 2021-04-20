""" QUESTO è per testare su un dataset interno allo ieo"""
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
ap.add_argument("-d", "--dataset",default=r"C:\Users\matte\Dropbox\fisica_medica\lavori_ieo\ml\radimetrics_test.csv")

arg=vars(ap.parse_args())

print(f"[INFO] Reading data from {arg['dataset']}")
X,y=data_to_model(pd.read_csv(arg["dataset"]))


print(X.mean())


# load the model from disk

filename=r"esperimenti\plain_rf\plain_rf.sav"
loaded_model = pickle.load(open(filename, 'rb'))
y_pred=loaded_model.predict(X)


data=pd.concat([X,y],axis=1)
data["prediction"]=y_pred
data.to_csv("test_processed.csv", index=False)
result = loaded_model.score(X, y)
print(result)
