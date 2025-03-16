import os
import sys
import pandas as pd
import numpy as np
from src.USA_Rain_Prediction_MLProject.logger import logging
from src.USA_Rain_Prediction_MLProject.exception import MyException
from src.USA_Rain_Prediction_MLProject.config import DataTransformationConfig
from src.USA_Rain_Prediction_MLProject.utils import save_object
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

class DataTransformation:
  def __init__(self):
    self.data_transformation_config = DataTransformationConfig()

  def get_data_transformation(self):
    try:
      numerical_features = ['Temperature', 'Humidity', 'Wind Speed', 
                     'Precipitation', 'Cloud Cover', 'Pressure']
      cat_features = ['Location']

      num_pipeline = Pipeline(steps=[
        ("scaler", StandardScaler(with_mean=False))
      ])
      cat_pipeline = Pipeline(steps=[
        ("encoder", OneHotEncoder()),
        ("scaler", StandardScaler(with_mean=False))
      ])

      preprocessor = ColumnTransformer([
        ("Numeric Pipeline", num_pipeline, numerical_features),
        ("Categorical Pipeline", cat_pipeline, cat_features)
      ])

      return preprocessor


    except Exception as e:
      raise MyException(e, sys)

  def initiate_data_transformation(self,train_path,test_path):
    try:
      train_df = pd.read_csv(train_path)
      test_df = pd.read_csv(test_path)
      preproccesor_obj = self.get_data_transformation()
      
      logging.info("Train and Test data reading")

      target_column = "Rain Tomorrow"

      input_feature_train_df = train_df.drop(columns=[target_column],axis=1)
      target_feature_train_df = train_df[target_column]

      input_feature_test_df = test_df.drop(columns=[target_column], axis=1)
      target_feature_test_df = test_df[target_column]

      input_feature_train_array=preproccesor_obj.fit_transform(input_feature_train_df).toarray()
      input_feature_test_array=preproccesor_obj.transform(input_feature_test_df).toarray()

      train_arr = np.c_[input_feature_train_array, np.array(target_feature_train_df)]
      test_arr = np.c_[input_feature_test_array, np.array(target_feature_test_df)]

      logging.info("Train and Test data transformation completed")

      save_object(
        file_path=self.data_transformation_config.data_transformation_path,
        obj=preproccesor_obj
      )

      

      logging.info("Preprocessor pickle file saved")
      return(
        train_arr,
        test_arr,
        self.data_transformation_config.data_transformation_path
      )
      
    except Exception as e:
      raise MyException(e, sys)

