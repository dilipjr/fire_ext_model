import sys
from pathlib import Path
import typing as t
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from fire_extinguish_model import __version__ as _version
from fire_extinguish_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config

def pre_pipeline_preparation(*, data_frame: pd.DataFrame) -> pd.DataFrame:
    
    data_frame = data_frame.fillna(0)  

    return data_frame

def _load_raw_dataset(*, file_name: str) -> pd.DataFrame:
    
    file_path = DATASET_DIR / file_name
    if file_path.is_file():
        dataframe = pd.read_csv(file_path)
        return dataframe
    raise FileNotFoundError(f"Dataset not found at {file_path!r}")

def load_dataset(*, file_name: str) -> pd.DataFrame:
   
    dataframe = _load_raw_dataset(file_name=file_name)
    transformed = pre_pipeline_preparation(data_frame=dataframe)

    return transformed

def save_pipeline(*, pipeline_to_persist: Pipeline) -> None:
   
    save_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name

    remove_old_pipelines(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)

def load_pipeline(*, file_name: str) -> Pipeline:
    
    file_path = TRAINED_MODEL_DIR / file_name
    if file_path.is_file():
        trained_model = joblib.load(filename=file_path)
        return trained_model
    raise FileNotFoundError(f"Pipeline not found at {file_path!r}")

def remove_old_pipelines(*, files_to_keep: t.List[str]) -> None:
    
    do_not_delete = files_to_keep + ["__init__.py"]
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()
