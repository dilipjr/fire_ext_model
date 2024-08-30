import sys
from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


from fire_extinguish_model.config.core import config
from fire_extinguish_model.pipeline import fire_extinguish_model_pipe
from fire_extinguish_model.processing.data_manager import load_dataset, save_pipeline

def run_training() -> None:
    
    data = load_dataset(file_name=config.app_config.training_data_file)
    
   
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.model_config.features],  
        data[config.model_config.target],    
        test_size=config.model_config.test_size,
        random_state=config.model_config.random_state,  
    )

    
    fire_extinguish_model_pipe.fit(X_train, y_train)
    
    
    y_pred = fire_extinguish_model_pipe.predict(X_test)

    
    print("Accuracy score:", accuracy_score(y_test, y_pred).round(2))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    
    save_pipeline(pipeline_to_persist=fire_extinguish_model_pipe)
    
if __name__ == "__main__":
    run_training()
