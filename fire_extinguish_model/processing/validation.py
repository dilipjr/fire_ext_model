import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from fire_extinguish_model.config.core import config
from fire_extinguish_model.processing.data_manager import pre_pipeline_preparation

def validate_inputs(*, input_df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
   
    pre_processed = pre_pipeline_preparation(data_frame=input_df)
    validated_data = pre_processed[config.model_config.features].copy()
    errors = None

    try:
        
        MultipleDataInputs(
            inputs=validated_data.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as error:
        errors = error.json()

    return validated_data, errors

class DataInputSchema(BaseModel):
    SIZE: Optional[str] 
    FUEL: Optional[str]   
    DISTANCE: Optional[str]  
    DESIBEL: Optional[str]  
    AIRFLOW: Optional[str]  
    FREQUENCY: Optional[str]  

class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]

