import os
import yaml
import sys
import pickle
import yaml
from src.exception import MyException
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV

def save_object(file_path, obj):
  try:
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)
    with open(file_path, "wb") as file_obj:
      pickle.dump(obj, file_obj) 
  except Exception as e:
    raise MyException(e, sys)
  
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise MyException(e,sys)
  
  
# def evaluate_models(X_train,y_train,X_test,y_test,models,params):
#   try:
#     report = {}
    
#     for model_name, model in models.items():  # Iterate directly over model names and objects
#         para = params.get(model_name, {})  # Get parameters for the current mode
#         rf = RandomizedSearchCV(model, para, cv=3, n_jobs=-1, verbose=2)
#         rf.fit(X_train, y_train)
#         # Update model with best parameters
#         best_model = model.set_params(**rf.best_params_)
#         best_model.fit(X_train, y_train)
#         # Predictions
#         y_test_pred = best_model.predict(X_test)
#         # Calculate accuracy
#         test_model_score = accuracy_score(y_test, y_test_pred)
#         # Store score in the report dictionary
#         report[model_name] = test_model_score
#     return report 


#   except Exception as e:
#     raise MyException(e, sys)
  
def read_yaml_file(file_path: str) -> dict:
  try:
      with open(file_path, "rb") as yaml_file:
          return yaml.safe_load(yaml_file)

  except Exception as e:
      raise MyException(e, sys) from e

