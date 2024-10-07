
import joblib
import json
import os
import pandas as pd

def init():
    global model
    model_dir = os.environ["AZUREML_MODEL_DIR"]
    model_path = os.path.join(model_dir, "best_model/model.pkl")
    model = joblib.load(model_path)

def run(data):
    try:
        input_data = json.loads(data)  
        columns = ["ram_num", "battery_power_num", "px_width_num", "px_height_num", 
                   "mobile_wt_num", "int_memory_num", "n_cores_cat"]
                   
        df = pd.DataFrame(input_data["input_data"]["data"], columns=columns)
        
        prediction = model.predict(df)
        
        return prediction.tolist() 
    
    except Exception as e:
        return {"error": str(e)}

