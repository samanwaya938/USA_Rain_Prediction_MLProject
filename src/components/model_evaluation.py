import os
import sys
import pandas as pd
from src.exception import MyException
from src.logger import logging
from src.utils import save_object
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV

class ModelEvaluation:
  def __init__(self):
    pass

  def evaluate_models(self,X_train,y_train,X_test,y_test,models,params):

    try:
      report = {}

      for model_name, model in models.items():  # Iterate directly over model names and objects
          para = params.get(model_name, {})  # Get parameters for the current mode
          rf = RandomizedSearchCV(model, para, cv=3, n_jobs=-1, verbose=2)
          rf.fit(X_train, y_train)
          # Update model with best parameters
          best_model = model.set_params(**rf.best_params_)
          best_model.fit(X_train, y_train)
          # Predictions
          y_test_pred = best_model.predict(X_test)
          # Calculate accuracy
          test_model_score = accuracy_score(y_test, y_test_pred)
          # Store score in the report dictionary
          report[model_name] = test_model_score
      return report 


    except Exception as e:
      raise MyException(e, sys)