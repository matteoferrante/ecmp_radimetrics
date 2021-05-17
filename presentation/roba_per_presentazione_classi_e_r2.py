import pandas as pd
from sklearn.metrics import r2_score,classification_report,confusion_matrix


data=pd.read_csv(r"test_processed.csv")

print(f"[R2]{r2_score(data['ICRP_103_mSv'],data['prediction_dosenet'])}")

def classify(x):
    x_class=[]
    for i in x:
        if i<1:
            v=1
        elif i>=1 and i<5:
            v=2
        elif i>=5 and i<10:
            v=3
        else:
            v=4
        x_class.append(v)
    return x_class

y_true=classify(data["ICRP_103_mSv"].values)
y_pred=classify(data['prediction_rf_finetuned'].values)


print(classification_report(y_true,y_pred))