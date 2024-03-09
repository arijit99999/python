import os
import sys
import pandas as pd
import numpy as np
from src.dimondpriceprediction.logger import logging
from src.dimondpriceprediction.exception import customexception
from dataclasses import dataclass
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler
from src.dimondpriceprediction.utlis.utils import save_object


@dataclass
class datatransform_config():
    preprocessor_path=os.oath.join("artifacts","preprocessor.pkl")

class datatransformation:
    def __init__(self):
        self.data_transformation_config=datatransform_config()

    def get_datatransformation(self):
        try:
            logging.info('data transformation initiated')
            categrical_cols=['cut', 'color', 'clarity']
            numerical_cols=['carat', 'depth', 'table', 'x', 'y', 'z', 'price']

            cut_category= ['Premium' 'Very Good' 'Ideal' 'Good' 'Fair']
            color_category= ['F' 'J' 'G' 'E' 'D' 'H' 'I']
            clarity_category= ['VS2' 'SI2' 'VS1' 'SI1' 'IF' 'VVS2' 'VVS1' 'I1']

            logging.info('pipeline  has been initiated')

            num_pipeline=Pipeline(
                steps=[
                    ('impute',SimpleImputer(strategy='median')),
                    ('scaling',StandardScaler())
                ]
            )
            cat_pipeline=Pipeline(
                steps=[
                    ('impute',SimpleImputer(strategy='most_frequent')),
                    ('odinal_encoder',OrdinalEncoder(categories=[cut_category,color_category,clarity_category])),
                    ('scaling',StandardScaler())
                ]
            )

            preprocessor=ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,numerical_cols),
                    ('cat_pipeline',cat_pipeline,categrical_cols)
                ]
            )

            return preprocessor


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
            
            preprocessor_obj=self.get_datatransformation()

            traget_col_name="price"
            drop_col=[traget_col_name,"id"]

            input_feature_for_train_data=train_d.drop(columns=drop_col,axis=1)
            target_feature_for_train_data=train_d[traget_col_name]
            input_feature_for_test_data=test_d.drop(columns=drop_col,axis=1)
            target_feature_for_test_data=train_d[traget_col_name]

            input_feature_for_train_data_preprocessor=preprocessor_obj.fit_transform(input_feature_for_train_data)
            input_feature_for_test_data_preprocessor=preprocessor_obj.transform(input_feature_for_test_data)

            logging.info('applied preprocessing object on train and test data')

            train_arr=np.c_[input_feature_for_train_data_preprocessor,np.array(target_feature_for_train_data)]
            test_arr=np.c_[input_feature_for_test_data_preprocessor,np.array(target_feature_for_test_data)]
            
            save_object(file_path=self.data_transformation_config.preprocessor_path,
                        obj=preprocessor_obj)

            logging.info('preprocessor file saved as pickle file')

            return(
                train_arr,
                test_arr
            )
        except Exception as e:
           logging.info("exception has been triggered at data transformation stage")
           raise customexception(e,sys)

