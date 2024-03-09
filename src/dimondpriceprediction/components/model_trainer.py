import os
import sys
import pandas as pd
from src.dimondpriceprediction.logger import logging
from src.dimondpriceprediction.exception import customexception
from dataclasses import dataclass
from sklearn.preprocessing import OrdinalEncoder,StandardScaler
from src.dimondpriceprediction.utlis.utils import save_object
from src.dimondpriceprediction.utlis.utils import evaluate_model

from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet


@dataclass
class model_training_config:
    train_model_path=os.path.join('artifacts','model.pkl')

class modeltraining:
    def __init__(self):
        self.model_train_config=model_training_config()
    def initiate_modeltraining(self):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')

            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            models={
            'LinearRegression':LinearRegression(),
            'Lasso':Lasso(),
            'Ridge':Ridge(),
            'Elasticnet':ElasticNet()
        }
            model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models)
            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')

            # To get best model score from dictionary 
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]

            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')

            save_object(
                 file_path=self.model_train_config.train_model_path,
                 obj=best_model
            )


        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise customexception(e,sys)