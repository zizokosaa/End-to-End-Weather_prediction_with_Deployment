from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import pickle
from pathlib import Path


class PredictionPipeline:
    def __init__(self):  # loading the model and scaler by default
        self.model = load_model(Path('artifacts/model_trainer/model.keras'))
        
        # Load the scaler used for TAVG column during training
        with open(Path('artifacts/model_trainer/tavg_scaler.pkl'), 'rb') as f:
            self.tavg_scaler = pickle.load(f)

    def predict(self, data):  # data shape: (1, 10, 9)
        scaled_prediction = self.model.predict(data)  # e.g., [[0.57]]

        # Inverse transform to original scale — ensure it's 2D
        original_prediction = self.tavg_scaler.inverse_transform(scaled_prediction)

        return original_prediction[0][0]  # return scalar value
