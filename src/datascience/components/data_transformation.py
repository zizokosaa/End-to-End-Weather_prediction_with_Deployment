import os
from src.datascience import logger
from sklearn.model_selection import train_test_split
import pandas as pd
from src.datascience.entity.config_entity import (DataTransformationConfig)

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def Preprocessing(self):
        data = pd.read_csv(self.config.data_path)
        data = data.drop(["STATION","TAVG_ATTRIBUTES","NAME"],axis= 1)

        data['DATE'] = pd.to_datetime(data['DATE'])
        data['YEAR'] = data['DATE'].dt.year
        data['MONTH'] = data['DATE'].dt.month
        data['DAY'] = data['DATE'].dt.day
        data['DAY_OF_WEEK'] = data['DATE'].dt.dayofweek
        data['DAY_OF_YEAR'] = data['DATE'].dt.dayofyear
        data = data.drop('DATE', axis = 1)
        data.to_csv(os.path.join(self.config.root_dir,"Temp_dataset_preprocessed.csv"),index = False)
        logger.info("Preprocessing the data")
        logger.info(data.shape)
        print(data.shape)