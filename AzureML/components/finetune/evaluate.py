
import argparse
import os
import pandas as pd
import mlflow
import numpy as np
from joblib import dump, load
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
import optuna

def select_first_file(path, num):
    files = os.listdir(path)
    return os.path.join(path, files[num])

def score(model, X_test, y_test):
    y_pred = model.predict(X_test)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    return precision,recall,f1
    
    
mlflow.sklearn.autolog()

os.makedirs("./outputs", exist_ok=True)

def main():
    """Main function for training the model."""
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", type=str, help="Path to test data")
    parser.add_argument("--model_1", type=str, help="Path to train data")
    parser.add_argument("--model_2", type=str, help="Path to train data")
    parser.add_argument("--model_3", type=str, help="Path to train data")
    parser.add_argument("--registered_model_name", type=str, help="Model name")
    parser.add_argument("--best_model", type=str, help="path to model file")
    args = parser.parse_args()
    

    model1 = load(select_first_file(args.model_1,1))
    model2 = load(select_first_file(args.model_2,1))
    model3 = load(select_first_file(args.model_3,1))

    test_df = pd.read_csv(select_first_file(args.test_data,0))

    X_test = test_df.drop(columns="price_range")
    y_test = test_df['price_range']

    precision_1,recall_1,f1_1 = score(model1, X_test, y_test)
    precision_2,recall_2,f1_2 = score(model2, X_test, y_test)
    precision_3,recall_3,f1_3 = score(model3, X_test, y_test)
    
    scores_df = pd.DataFrame({'model': ['model1', 'model2', 'model3'],
                                'precision': [precision_1, precision_2, precision_3],
                                'recall': [recall_1, recall_2, recall_3],
                                'f1': [f1_1, f1_2, f1_3]})

    sorted_scores_df = scores_df.sort_values(by='f1', ascending=False)

    Best_model = sorted_scores_df.iloc[0]['model']
    best_f1score = sorted_scores_df.iloc[0]['f1']

    if Best_model == 'model1':
        Best_model = model1
    elif Best_model == 'model2':
        Best_model = model2
    else:
        Best_model = model3

    results_path = os.path.join(args.best_model, "test_result.csv")
    sorted_scores_df.to_csv(results_path, index=False)
    mlflow.log_artifact(results_path) 

    if mlflow.active_run():
        mlflow.end_run()
        
    with mlflow.start_run(run_name=args.registered_model_name):
        print("Registering the model via MLFlow")
        mlflow.sklearn.log_model(
            sk_model=Best_model,
            registered_model_name=args.registered_model_name,
            artifact_path='best_model',  
        )

    mlflow.end_run()


if __name__ == "__main__":
    main()
