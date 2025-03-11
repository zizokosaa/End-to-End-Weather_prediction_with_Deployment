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
                
        return data


        
    def train_test_splitting(self, data: pd.DataFrame):
        train, test = train_test_split(data)
        train.to_csv(os.path.join(self.config.root_dir,"train.csv"),index=False)
        test.to_csv(os.path.join(self.config.root_dir,"test.csv"),index = False)

        logger.info("Splited data into training and test sets")
        logger.info(train.shape)
        logger.info(test.shape)

        print(train.shape)
        print(test.shape)