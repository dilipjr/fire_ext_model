import sys
from pathlib import Path



from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier  

from fire_extinguish_model.config.core import config
from fire_extinguish_model.processing.features import LabelEncoderCustom


fire_extinguish_model_pipe = Pipeline([

    
    
    ('encode_fuel', LabelEncoderCustom(variable='FUEL')), 

    
    ('scaler', StandardScaler()),

    
    ('model_xgb', XGBClassifier(n_estimators=100, max_depth=10, random_state=42))
])