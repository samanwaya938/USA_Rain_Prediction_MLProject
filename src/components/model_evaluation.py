import os
import sys
import mlflow
import dagshub
import numpy as np
import tempfile
from src.exception import MyException
from src.logger import logging
from src.utils import save_object
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import RandomizedSearchCV

class ModelEvaluation:
  def __init__(self):        
    dagshub.init(repo_owner='samanwaya938', repo_name='USA_Rain_Prediction_MLProject', mlflow=True)

    mlflow.set_tracking_uri("https://dagshub.com/samanwaya938/USA_Rain_Prediction_MLProject.mlflow") # Add if using a remote server
    mlflow.set_experiment("Dagshub_ML_Experiment")
    logging.info("Initialized Dagshub Experiment tracking")

  def evaluate_models(self,X_train,y_train,X_test,y_test,models,params):

    try:
      report = {}

      for model_name, model in models.items():  # Iterate directly over model names and objects
          para = params.get(model_name, {})  # Get parameters for the current mode
          # print(f"\n{'='*40}")
          # print(f"Processing model: {model_name}")
          # print(f"Parameters received: {para}")

          # if not para:              
          #     print(f"⚠️ No parameters found for {model_name} in params.yaml")


          rf = RandomizedSearchCV(model, para, cv=3, n_jobs=-1, verbose=2)
          rf.fit(X_train, y_train)
          print(f"\n{model_name} Best Parameters:")
          print(rf.best_params_)

          # Update model with best parameters
          best_model = model.set_params(**rf.best_params_)
          best_model.fit(X_train, y_train)
          # Predictions
          y_test_pred = best_model.predict(X_test)
          # Confusion Matrix
          cm = confusion_matrix(y_test, y_test_pred)
          # Classification Report
          clf_report = classification_report(y_test, y_test_pred)
          # Calculate accuracy
          test_model_score = accuracy_score(y_test, y_test_pred)
          # Store score in the report dictionary
          report[model_name] = test_model_score

          with mlflow.start_run(run_name=model_name):                    
                    mlflow.log_params(rf.best_params_)                    
                    mlflow.log_metric("accuracy", test_model_score)

                    # Log confusion matrix and classification report as artifacts
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        cm_path = os.path.join(tmp_dir, "confusion_matrix.txt")
                        np.savetxt(cm_path, cm, fmt='%d')
                        mlflow.log_artifact(cm_path, "confusion_matrix")

                        cr_path = os.path.join(tmp_dir, "classification_report.txt")
                        with open(cr_path, 'w') as f:
                            f.write(clf_report)
                        mlflow.log_artifact(cr_path, "classification_report")

                    # Log the trained model
                    mlflow.sklearn.log_model(sk_model=best_model,
                        artifact_path=model_name,
                        registered_model_name=model_name)
      return report, cm, clf_report


    except Exception as e:
      raise MyException(e, sys)