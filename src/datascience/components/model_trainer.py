import pandas as pd
import os
from src.datascience import logger
from src.datascience.entity.config_entity import ModelTrainerConfig
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def create_sequence_and_training(self):
        scaler = MinMaxScaler()
        df = pd.read_csv(self.config.data_path)

        scaled_data = scaler.fit_transform(df)

        sequence_length = self.config.sequence_length
        num_features = len(df.columns)

        sequences = []
        labels = []
        for i in range(len(scaled_data) - sequence_length):
            seq = scaled_data[i:i+sequence_length]
            label = scaled_data[i+sequence_length][3]
            sequences.append(seq)
            labels.append(label)

        sequences = np.array(sequences)
        labels = np.array(labels)

        train_size = int(0.8 * len(sequences))
        train_x, test_x = sequences[:train_size], sequences[train_size:]
        train_y, test_y = labels[:train_size], labels[train_size:]

        print("Train X shape:", train_x.shape)
        print("Train Y shape:", train_y.shape)
        print("Test X shape:", test_x.shape)
        print("Test Y shape:", test_y.shape)
        return train_x,train_y

    def Creating_model(self,train_x,train_y):
        model = Sequential([
            Input(shape=(train_x.shape[1], train_x.shape[2])),
            LSTM(units=128, return_sequences=True),
            Dropout(0.2),
            LSTM(units=64, return_sequences=True),
            Dropout(0.2),
            LSTM(units=32, return_sequences=False),
            Dropout(0.2),
            Dense(units=1)
        ])

        # Compile the model
        model.compile(optimizer=self.config.optimizer, loss='mean_squared_error')
        early_stopping = EarlyStopping(monitor='val_loss', patience=self.config.patience, restore_best_weights=True)
        model_checkpoint = ModelCheckpoint(os.path.join(self.config.root_dir, self.config.model_name), monitor='val_loss', save_best_only=True)

        # Train the model
        history = model.fit(
            train_x, train_y,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            validation_split=0.2,  # Use part of the training data as validation
            callbacks=[early_stopping, model_checkpoint])


    