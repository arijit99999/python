import os
import sys
import pandas as pd
from src.dimondpriceprediction.logger import logging
from src.dimondpriceprediction.exception import customexception
from dataclasses import dataclass
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler


@dataclass
class datatransform_config():
    preprocessor_path=os.oath.join("artifacts","preprocessor.pkl")

class datatransformation:
    def __init__(self):
        self.data_transformation_config=datatransform_config()

    def get_datatransformation(self):
        try:
            pass
        except Exception as e:
           logging.info("exception has been triggered at data transformation stage")
           raise customexception(e,sys)

    def initalize_data_transformation(self,train_path,test_path):


        try:
            train_d=pd.read_csv(train_path)
            test_d=pd.read_csv(test_path)
            logging.info("reading of test and traning data is complited")
            logging.info(f'train dataframe head : \n{train_d.head().to_string()}')
            logging.info(f'test dataframe head : \n{test_d.head().to_string()}')


        except Exception as e:
           logging.info("exception has been triggered at data transformation stage")
           raise customexception(e,sys)

