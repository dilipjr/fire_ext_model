from typing import Any, List, Optional

from pydantic import BaseModel
from fire_extinguish_model.processing.validation import DataInputSchema


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    predictions: Optional[List[int]]
    #predictions: Optional[int]


class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                        "SIZE": 1,
                        "FUEL": "gasoline",
                        "DISTANCE": 10,
                        "DESIBEL": 96,
                        "AIRFLOW": 0,
                        "FREQUENCY": 72,
                        
                    }
                ]
            }
        }
