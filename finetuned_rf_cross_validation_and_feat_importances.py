"""USING BEST PARAMETERS FOR CROSS VALIDATION OF RANDOM FOREST"""

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
report=open(r"esperimenti\plain_rf\finetuned_cv_rf", "w")

#load data

print(f"[INFO] Reading data from {arg['dataset']}")
X,y=data_to_model(pd.read_csv(arg["dataset"]))

## PLAIN RANDOM FOREST

report.write("ESPERIMENTO 3. FINE TUNED RANDOMFOREST REGRESSOR:\n")
report.write("\t\t Dati non riscalati, best paramaters\n\n")


scoring = { 'r2': 'r2',"explained_variance_score":'explained_variance',"max error":'max_error'}
#scoring=make_scorer(explained_variance_score,max_error,mean_absolute_error,r2_score)
regr=RandomForestRegressor( max_features='sqrt',  n_estimators=1000)
scores= cross_validate(regr, X, y, cv=10,n_jobs=-1,verbose=1)

print(scores)

report.write(f"10 fold-cross validation: \n{scores}\n")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

print(f"[INFO] Fitting model")
regr.fit(X_train,y_train)

y_pred=regr.predict(X_test)



print(f"Fitted r2 model score: {regr.score(X_test,y_test)}")

report.write(f"Fitted r2 model score: {regr.score(X_test,y_test)}\n")




# MEAN ABSOLUTE ERRROR
mae=mean_absolute_error(y_test,y_pred)
print(f"mean_absolute_error score: {mae}")

report.write(f"mean_absolute_error score: {mae}\n")


# MAX ERROR
max_er=max_error(y_test,y_pred)
print(f"max_error score: {max_er}")

report.write(f"max_error score: {max_er}\n")



filename = r'esperimenti\plain_rf\finetuned_last_rf.sav'
pickle.dump(regr, open(filename, 'wb'))
print(f"[INFO] Model saved")



### FEATURE IMPORTANCES

fi_vals=regr.feature_importances_
fi_names=X.columns

features=dict(zip(fi_names,fi_vals))
features=dict(sorted(features.items(),key=lambda x:x[1],reverse=True))

print(features)
report.write(f"feature importances: {features}\n")



### LAVORARE QUA PER FARE BENE L'IMMAGINE!!


fig,ax=plt.subplots(1)

#ax.bar(features.keys(),features.values())




df=pd.DataFrame.from_dict([features])
sns.set_theme(style="whitegrid")
sns.barplot( data=df,ax=ax)

ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.tight_layout()
plt.savefig(r'esperimenti\plain_rf\finetuned_features.png')