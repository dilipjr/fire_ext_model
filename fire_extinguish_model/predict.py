import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))


from typing import Union
import pandas as pd
import numpy as np

from fire_extinguish_model import __version__ as _version
from fire_extinguish_model.config.core import config
from fire_extinguish_model.processing.data_manager import load_pipeline
from fire_extinguish_model.processing.data_manager import pre_pipeline_preparation
from fire_extinguish_model.processing.validation import validate_inputs


pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
fire_ext_pipe = load_pipeline(file_name=pipeline_file_name)

def make_prediction(*, input_data: Union[pd.DataFrame, dict]) -> dict:
   
    
    validated_data, errors = validate_inputs(input_df=pd.DataFrame(input_data))
    
    
    validated_data = validated_data.reindex(columns=config.model_config.features)
    
    results = {"predictions": None, "version": _version, "errors": errors}
      
    if not errors:
        predictions = fire_ext_pipe.predict(validated_data)
        results = {"predictions": predictions.tolist(), "version": _version, "errors": errors}
        print(results)

    return results

if __name__ == "__main__":
    
    data_in = {
        'SIZE': [10],
        'FUEL': ['thinner'],
        'DISTANCE': [70],
        'DESIBEL': [75],
        'AIRFLOW': [50],
        'FREQUENCY': [750]
    }

    make_prediction(input_data=data_in)
