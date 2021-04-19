""" Questo codice fa tutto il preprocessing dei dati in modo da restituire subito i dati pronti"""
from sklearn.preprocessing import LabelEncoder

from Preprocess.RadimetricsPreprocessor import RadimetricsPreprocessor



def prepare_dataset(data):
    pre=RadimetricsPreprocessor(data)


    pre.basic_filter()   # basic filter -> rimuove indici e calcola l'età

    #definisco colonne da non usare

    to_drop=['Device',"Acquisition_Type","Acquisition_Protocol_Name","Series_Description","mAs_Modulated","CTDIvol_Head_Max_mGy","CTDIvol_Body_Max_mGy"]
    pre.drop_columns(to_drop)

    extra_drop=["Description"] #provo a eliminare anche la description
    pre.drop_columns(extra_drop)



    #nan removing
    subset=['Age', 'Weight','Height','Gender','Filter_Type', "ICRP_103_mSv"]

    pre.dropna(subset=subset)
    data=pre.data

    #per le colonne del CTDI e del DLP sostituisco i nan con zero

    fill_subset=['CTDIvol_Body_mGy', 'CTDIvol_Head_mGy', 'DLP_Body_mGy_cm',
           'DLP_Head_mGy_cm',]

    #per il pitch sostituisco con 1.
    fill_one_subset=["Pitch"]
    pre.fillna(subset=fill_subset,inpute=0.)
    pre.fillna(subset=fill_one_subset,inpute=1.)
    return data

def encode_label(data):
    label_bins = {}
    for c in data.columns:

        if data[c].dtype == object:
            print(f"[INFO] binarizing {c}")
            n = data[c].values
            lb = LabelEncoder()
            data[c] = lb.fit_transform(data[c].values)
            label_bins[c] = lb
    return data,label_bins


def data_to_model(data):


    data=prepare_dataset(data)

    ### ENCODING
    # qua devo fare l'encoding di Gender e Filter

    data,label_bins=encode_label(data)


    #preparo X e y
    X=data.drop("ICRP_103_mSv",axis=1)
    y=data["ICRP_103_mSv"]
    return X,y