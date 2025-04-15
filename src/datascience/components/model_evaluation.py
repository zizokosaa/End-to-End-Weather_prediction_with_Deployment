import os
import pandas as pd
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from urllib.parse import urlparse
import mlflow
import mlflow.keras
import numpy as np
from tensorflow.keras.models import load_model
from src.datascience.entity.config_entity import ModelEvaluationConfig
from src.datascience.utils.common import save_json
from pathlib import Path




os.environ['MLFLOW_TRACKING_URI'] = "https://dagshub.com/zizokosaa/datascienceproject.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME'] = "zizokosaa"
os.environ['MLFLOW_TRACKING_PASSWORD'] = "0a814d4b710305469e1d136c3e175e425874d6af"


class ModelEvaluation:
    def __init__(self,config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self,actual,pred):
        rmse = np.sqrt(mean_squared_error(actual,pred))
        mae = mean_absolute_error(actual,pred)
        r2 = r2_score(actual,pred)
        return rmse, mae, r2
    
    def log_into_mlflow(self,test_x,test_y):
        model = load_model(self.config.model_path)

        
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            predicted_qualities = model.predict(test_x)

            (rmse,mae,r2) = self.eval_metrics(test_y,predicted_qualities)

            # Saving metrics as local
            scores = {"rmse": rmse, "mae": mae,"r2":r2}
            save_json(path = Path(self.config.metric_file_name),data=scores)

            mlflow.log_params(self.config.all_params)

            mlflow.log_metric("rmse",rmse)
            mlflow.log_metric("mae",mae)
            mlflow.log_metric("r2",r2)


            # Model registry does not work with file store
            if tracking_url_type_store != "file":
                mlflow.keras.log_model(
                    model,
                    "model",
                    registered_model_name="LSTMModel"
                )
            else:
                mlflow.keras.log_model(model, "model")