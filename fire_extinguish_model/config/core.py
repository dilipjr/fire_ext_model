import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from pathlib import Path
from typing import Dict, List 

from pydantic import BaseModel
from strictyaml import YAML, load

import fire_extinguish_model

# testing git push

PACKAGE_ROOT = Path(fire_extinguish_model.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"


DATASET_DIR = PACKAGE_ROOT / "datasets"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"


class AppConfig(BaseModel):
    
    package_name: str
    training_data_file: str
    pipeline_name: str
    pipeline_save_file: str

class ModelConfig(BaseModel):
   
    target: str  
    features: List[str]  
    
    
    
    
    test_size: float  
    random_state: int  
    n_estimators: int  
    max_depth: int  

class Config(BaseModel):
    
    app_config: AppConfig
    model_config: ModelConfig

def find_config_file() -> Path:
    
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f"Config not found at {CONFIG_FILE_PATH!r}")

def fetch_config_from_yaml(cfg_path: Path = None) -> YAML:
    
    if not cfg_path:
        cfg_path = find_config_file()
    if cfg_path:
        with open(cfg_path, "r") as conf_file:
            parsed_config = load(conf_file.read())
            return parsed_config
    raise OSError(f"Did not find config file at path: {cfg_path}")

def create_and_validate_config(parsed_config: YAML = None) -> Config:
    
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()
    
   
    _config = Config(
        app_config = AppConfig(**parsed_config.data),
        model_config = ModelConfig(**parsed_config.data),
    )
    return _config

config = create_and_validate_config()