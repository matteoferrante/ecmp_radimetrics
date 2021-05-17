
import argparse,os
import pandas as pd
from Preprocess.data_to_model import data_to_model

ap=argparse.ArgumentParser()
ap.add_argument("-d", "--dataset",default=r"C:\Users\matte\Dropbox\fisica_medica\lavori_ieo\ml\radimetrics_train.csv")

arg=vars(ap.parse_args())




os.makedirs(r"esperimenti\plain_rf",exist_ok=True)
report=open(r"esperimenti\plain_rf\report_tuned_rf", "w")

#load data

print(f"[INFO] Reading data from {arg['dataset']}")
X,y=data_to_model(pd.read_csv(arg["dataset"]))

X["ICRP_103_mSv"]=y
X.to_csv("internal_training_preprocessed.csv")