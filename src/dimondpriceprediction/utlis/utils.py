import os
import sys
import pickle
import pandas as pd
import numpy as np
from src.dimondpriceprediction.logger import logging
from src.dimondpriceprediction.exception import customexception



def save_object(file_path, obj):
    try:
        pickle.dump(obj,open(file_path,"wb"))

    except Exception as e:
        raise customexception(e, sys)
