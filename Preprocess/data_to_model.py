""" Questo codice fa tutto il preprocessing dei dati in modo da restituire subito i dati pronti"""
from sklearn.preprocessing import LabelEncoder

from Preprocess.RadimetricsPreprocessor import RadimetricsPreprocessor



def prepare_dataset(data,sep="-"):
    pre=RadimetricsPreprocessor(data)


    pre.basic_filter(sep=sep)   # basic filter -> rimuove indici e calcola l'età

    #definisco colonne da non usare

    to_drop=['Device',"Acquisition_Type","Acquisition_Protocol_Name","Series_Description","mAs_Modulated","CTDIvol_Head_Max_mGy","CTDIvol_Body_Max_mGy"]
    try:
        pre.drop_columns(to_drop)
    except:
        print(f"[INFO] could not remove those columns.")

    extra_drop=["Description",'Filter_Type'] #provo a eliminare anche la description
    try:
        pre.drop_columns(extra_drop)
    except:
        print(f"[INFO] could not remove Description. Maybe it's already been removed.")


    #nan removing
    subset=['Age', 'Weight','Height','Gender', "ICRP_103_mSv","kVp","Nominal_mA"]

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
    #label_bins = {}
    for c in data.columns:

        if data[c].dtype == object:
            print(f"[INFO] binarizing {c}")
            #n = data[c].values
            #lb = LabelEncoder()
            data[c] = data[c].map({"M":0,"F":1,"O":0})  #rimappo gli other come M
            #label_bins[c] = lb
    return data


def data_to_model(data,sep="-",thr=None):


    data=prepare_dataset(data,sep=sep)

    ### ENCODING
    # qua devo fare l'encoding di Gender e Filter

#    data,label_bins=encode_label(data)

    data=encode_label(data)

    #se presente soglia la applico
    if thr is not None:
        data=data[data["ICRP_103_mSv"]>=thr]


    print(f"[INFO] At the end of pre-processing the output is composed by {len(data)} samples")
    #preparo X e y
    X=data.drop("ICRP_103_mSv",axis=1)
    y=data["ICRP_103_mSv"]
    return X,y