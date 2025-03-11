## component-Data Ingestion
import os
import urllib.request as request
from src.datascience import logger
import zipfile
from src.datascience.entity.config_entity import (DataIngestionConfig)
import shutil

class DataIngestion:
    def __init__(self,config: DataIngestionConfig):
        self.config = config
    
    # Downloading the zip file
    def copy_data_file(self):
        """
        Copies the file "Temp_dataset.csv" from config.source_URL to config.local_data_file.
        Function returns None
        """
        source_file_path = self.config.source_URL  # Path to the downloaded file
        local_file_path = self.config.local_data_file  # Path to save the copy

        # Ensure the directory for the local file exists
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

        # Copy the file
        shutil.copy(source_file_path, local_file_path)
        logger.info(f"File copied from {source_file_path} to {local_file_path}")