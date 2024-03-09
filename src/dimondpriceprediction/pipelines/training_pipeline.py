from src.dimondpriceprediction.components.data_ingestion import DataIngestion
import pandas as pd
import numpy as np
import os
import sys
from src.dimondpriceprediction.logger import logging
from src.dimondpriceprediction.exception import customexception

obj=DataIngestion()
obj.initiate_data_ingestion()