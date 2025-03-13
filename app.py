from src.USA_Rain_Prediction_MLProject.logger import logging
from src.USA_Rain_Prediction_MLProject.exception import MyException
import sys

if __name__ == "__main__":
  logging.info("logging has started")
  try:
    1/0
  except Exception as e:
    raise MyException(e, sys)
