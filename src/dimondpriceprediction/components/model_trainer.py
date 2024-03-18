
import pandas as pd
import numpy as np
import os
import sys
from src.dimondpriceprediction.logger import logging
from src.dimondpriceprediction.exception import customexception
from dataclasses import dataclass
from src.dimondpriceprediction.utlis.utils import save_object
from sklearn.metrics import r2_score

from sklearn.linear_model import LinearRegression, Ridge,Lasso,ElasticNet

 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')
    
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initate_model_training(self,train_array,test_array):
        try:
            x_train=train_array[:,:-1]
            y_train=train_array[:,-1:]
            x_test=test_array[:,:-1]
            y_test=test_array[:,-1:]
            logging.info('read x_train,x_test,y_train,y_test')
            model=LinearRegression()
            model.fit(x_train,y_train)
            y_pred=model.predict(x_test)
            score=r2_score(y_test,y_pred)
            logging.info(f'model score is {score*100} %')
            print(round(r2_score(y_test,y_pred)*100,2),'%')
            save_object(obj=model,file_path=self.model_trainer_config.trained_model_file_path)
            logging.info('model traing has been complited and we have saved the model')
            return self.model_trainer_config.trained_model_file_path
        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise customexception(e,sys)

        
    