from dataclasses import dataclass
from pathlib import Path

@dataclass # we don't need to use self keyword
class DataIngestionConfig:
    root_dir: Path
    source_URL: Path
    local_data_file: Path

@dataclass
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    unzip_data_dir: Path
    all_schema: dict

@dataclass
class DataTransformationConfig:
    root_dir: Path
    data_path: Path

@dataclass
class ModelTrainerConfig:
    root_dir: Path
    data_path: Path
    model_name: str
    sequence_length: int
    optimizer: str
    learning_rate: float
    batch_size: int
    epochs: int
    patience: int
    target_column: str

@dataclass
class ModelEvaluationConfig:
    root_dir: Path
    test_data_path: Path
    model_path: Path
    all_params: dict
    metric_file_name: Path
    target_column: str
    mlflow_uri: str