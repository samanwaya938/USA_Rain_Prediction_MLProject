import sys
from src.USA_Rain_Prediction_MLProject.exception import MyException
from src.USA_Rain_Prediction_MLProject.components.data_ingestion import DataIngestion
from src.USA_Rain_Prediction_MLProject.components.data_transformation import DataTransformation
from src.USA_Rain_Prediction_MLProject.components.model_training import ModelTrainer

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