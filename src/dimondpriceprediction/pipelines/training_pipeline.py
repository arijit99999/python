from src.dimondpriceprediction.components.data_ingestion import DataIngestion

from src.dimondpriceprediction.components.data_transformation import DataTransformation

from src.dimondpriceprediction.components.model_trainer import ModelTrainer

import os
import sys
from src.dimondpriceprediction.logger import logging
from src.dimondpriceprediction.exception import customexception
import pandas as pd

obj=DataIngestion()
train_data_path,test_data_path=obj.initiate_data_ingestion()



data_transformatio=DataTransformation()
train_arr,test_arr=data_transformatio.initialize_data_transformation(train_data_path,test_data_path)


model_trainer_obj=ModelTrainer
model_trainer_obj.initate_model_training(train_arr,test_arr)

