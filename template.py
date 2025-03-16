import os
from pathlib import Path
import logging

project_name = "USA_Rain_Prediction_MLProject"
list_of_files = [
  ".github/workflows/.gitkeep",
  "Data/"
  f"src/{project_name}/__init__.py",
  f"src/{project_name}/components/__init__.py",
  f"src/{project_name}/utils/__init__.py",
  f"src/{project_name}/logger/__init__.py",
  f"src/{project_name}/exception/__init__.py",
  f"src/{project_name}/config/__init__.py",
  f"src/{project_name}/pipeline/__init__.py",
  f"src/{project_name}/pipeline/training_pipeline.py",
  f"src/{project_name}/pipeline/prediction_pipeline.py",
  "params.yaml",
  "app.py",
  "main.py"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory; {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")