import yaml
import os
import sys
from src.exception import MyException
from src.logger import logging
from src.config import ModelTrainerConfig
from src.utils import save_object, read_yaml_file
from src.components.model_evaluation import ModelEvaluation
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

class ModelTrainer:
  def __init__(self):
    self.model_trainer_config = ModelTrainerConfig()
    self.model_eval = ModelEvaluation()
   

  def initiate_model_trainer(self,train_arr,test_arr):
    
    try:
      logging.info("Splitting training and test data")
      X_train, y_train, X_test, y_test = train_arr[:,:-1], train_arr[:,-1], test_arr[:,:-1], test_arr[:,-1]

      models = {
      "Logistic regression" : LogisticRegression(),
      "Decision Tree" : DecisionTreeClassifier(),
      "Support Vector Machine" : SVC(),
      "Random Forest" : RandomForestClassifier(),
      "K-Nearest Neighbors" : KNeighborsClassifier()
      }

      params = read_yaml_file(os.path.join("params.yaml"))

    

      model_report:dict = self.model_eval.evaluate_models(X_train,y_train,X_test,y_test,models,params)
      print(f"Model Report : {model_report}")

      logging.info("Model training completed")

      best_model_score = max(sorted(model_report.values()))
      print(f"Best Model Score : {best_model_score}")
      
      if best_model_score < 0.6:
        raise Exception("No best model found")

      best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
      best_model = models[best_model_name]

      save_object(
        file_path=self.model_trainer_config.model_training_path,
        obj=best_model
      )

      return (
        self.model_trainer_config.model_training_path
      )

    except Exception as e:
      raise MyException(e, sys)