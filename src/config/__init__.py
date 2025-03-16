import os
from dataclasses import dataclass
@dataclass
class DataIngestionConfig:
  train_data_path:str = os.path.join('artifact','data_ingestion','train.csv')
  test_data_path:str = os.path.join('artifact','data_ingestion','test.csv')
  raw_data_path:str = os.path.join('artifact','data_ingestion','raw.csv')

@dataclass
class DataTransformationConfig:
  data_transformation_path:str = os.path.join('artifact', 'Data_transformation', 'preprossed.pkl')

class ModelTrainerConfig:
  model_training_path:str = os.path.join('artifact', 'model_trainer', 'model.pkl')
  