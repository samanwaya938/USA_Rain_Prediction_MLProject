import sys
from src.exception import MyException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTrainer

if __name__ == "__main__":
  try:
    data_ingestion = DataIngestion()
    train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

    model_trainer = ModelTrainer()
    model_trainer.initiate_model_trainer(train_arr, test_arr)

  except Exception as e:
    raise MyException(e, sys)