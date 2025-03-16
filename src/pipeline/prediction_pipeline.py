import sys
import os
import pandas as pd
from src.exception import MyException
from src.logger import logging
from src.utils import load_object

class PredictionPipeline:
  def __init__(self):
    pass

  def predict(self,features):
    try:      
      preprocessor_path = os.path.join("artifact","Data_transformation","preprossed.pkl")
      model_path = os.path.join("artifact","model_trainer","model.pkl")

      preprocessor = load_object(preprocessor_path)
      model = load_object(model_path)

      data_scaled = preprocessor.transform(features)
      prediction = model.predict(data_scaled)

      return prediction
    except Exception as e:
      raise MyException(e,sys)
    
class CustomData:
  def __init__(self,
               date: str,
               location: str,
               temperature: float,
               humidity: float,
               wind_speed: float,
               precipitation: float,
               cloud_cover: float,
               pressure: float,
              
               ):
    self.date = date
    self.location = location
    self.temperature = temperature
    self.humidity = humidity
    self.wind_speed = wind_speed
    self.precipitation = precipitation
    self.cloud_cover = cloud_cover
    self.pressure = pressure

  def get_data_as_dataframe(self):
    try:
      custom_data_input_dict = {
        "Date": [self.date],
        "Location": [self.location],
        "Temperature": [self.temperature],
        "Humidity": [self.humidity],
        "Wind Speed": [self.wind_speed],
        "Precipitation": [self.precipitation],
        "Cloud Cover": [self.cloud_cover],
        "Pressure": [self.pressure]
      }

      return pd.DataFrame(custom_data_input_dict)

    except Exception as e:
      raise MyException(e,sys)