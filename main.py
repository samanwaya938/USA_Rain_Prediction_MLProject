import sys
from src.USA_Rain_Prediction_MLProject.exception import MyException
from src.USA_Rain_Prediction_MLProject.components.data_ingestion import DataIngestion

if __name__ == "__main__":
  try:
    data_ingestion = DataIngestion()
    data_ingestion.initiate_data_ingestion()
  except Exception as e:
    raise MyException(e, sys)