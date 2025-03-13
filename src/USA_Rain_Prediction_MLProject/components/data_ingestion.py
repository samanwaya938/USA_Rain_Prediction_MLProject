import os
import sys
from src.USA_Rain_Prediction_MLProject.logger import logging
from src.USA_Rain_Prediction_MLProject.exception import MyException
from src.USA_Rain_Prediction_MLProject.config import DataIngestionConfig
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass



class DataIngestion:
  def __init__(self):
    self.data_ingestion_config = DataIngestionConfig()

  def initiate_data_ingestion(self):
    try:
      df = pd.read_csv(r'Data\usa_rain_prediction.csv')
      logging.info('Read the dataset as dataframe')
      os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path), exist_ok=True)
      df.to_csv(self.data_ingestion_config.raw_data_path,index=False)
      
      logging.info("Train test split initiated")
      train_df, test_df = train_test_split(df, random_state=42, test_size=0.2)
      train_df.to_csv(self.data_ingestion_config.train_data_path,index=False)
      test_df.to_csv(self.data_ingestion_config.test_data_path,index=False)

      logging.info("Ingestion of data is completed")

      return(
        self.data_ingestion_config.train_data_path,
        self.data_ingestion_config.test_data_path
      )

    except Exception as e:
      raise MyException(e, sys)
