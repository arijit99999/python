import os
import sys
import pickle
import pandas as pd
import numpy as np
from src.dimondpriceprediction.logger import logging
from src.dimondpriceprediction.exception import customexception

from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

def save_object(file_path,obj):
    try:
        dir_path=os.path.join(file_path)

        os.makedirs(dir_path,exist_ok=)

        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)
    except Exception as e:
        logging.info("exception has been triggered at utils stage")
        raise customexception(e,sys)
    
def evaluate_model(x_train,y_train,x_test,y_test,models):
    try:
        report={}
        
    except Exception as e:
        logging.info("exception has been triggered utlis stage")
        raise customexception(e,sys)
