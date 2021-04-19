"""Questa è una classe per preprocessare i dati esportati da radimetric"""
from datetime import date, datetime

import pandas as pd
from dateutil.relativedelta import relativedelta
import numpy as np

class RadimetricsPreprocessor:


    def __init__(self,data):
        """

        :param data: dataframe
        """
        self.data=data


    def basic_filter(self,calc_age=True,check_height=True):
        print(f"[INFO] Performing first filtration")
        if "Unnamed: 0" in self.data.columns:
            print(f"\t\t -Drop index column")
            self.data.drop(["Unnamed: 0"], axis=1, inplace=True)

        if calc_age:
            print(f"\t\t -Age Calculation\n")
            self.data["Age"] = self.data["DOB"].apply(self.calc_age)
            # drop the birthday column and all the rows where age is nan
            self.data.drop(["DOB"], axis=1, inplace=True)
        if check_height:
            self.check_height()

        return self.data

    def check_height(self):

        if self.data.Height.mean()<100:
            print(f"\t\t- Height seems to be in m. Converting to cm")
            self.data.Height=self.data.Height*100


    def calc_age(self,string):
        today = date.today()
        try:
            s = string.split("-")
            f = datetime(int(s[0]), int(s[1]), int(s[2]))
            time_difference = relativedelta(today, f)
            return int(time_difference.years)

        except Exception as e:
            return np.nan

    def drop_columns(self,to_drop):
        print(f"[INFO] Dropping columns: {to_drop}")
        self.data.drop(to_drop,axis=1,inplace=True)
        return self.data


    def dropna(self,subset):

        print(f"[INFO] Dropping Nan in {subset}")

        self.data.dropna(
            subset=subset,inplace=True)
        return self.data


    def fillna(self,subset,inpute=0.):

        print(f"[INFO] Inputing Nan in {subset}")

        for c in subset:
            self.data[c].fillna(0, inplace=True)
        return self.data