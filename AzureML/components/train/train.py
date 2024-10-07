
import argparse
import os
import pandas as pd
import mlflow
import numpy as np

from joblib import dump
from joblib import load

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import f1_score
import mltable

def select_first_file(path):
    files = os.listdir(path)
    return os.path.join(path, files[0])
    
def evaluate_model(scaler, X_train, y_train, num_features, cat_features, model, cv):
    transformer = ColumnTransformer(transformers=[('num', scaler, num_features), ('cat', OneHotEncoder(), cat_features)])
    pipeline = Pipeline([('preprocessor', transformer), ('model', model)])

    ave_precision = np.mean(cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='precision_weighted'))
    ave_recall = np.mean(cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='recall_weighted'))
    ave_f1 = np.mean(cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1_weighted'))
    ave_accuracy = np.mean(cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='accuracy'))
    
    return ave_precision, ave_recall, ave_f1, ave_accuracy

# Start Logging
mlflow.start_run()

# Enable autologging
mlflow.sklearn.autolog()

os.makedirs("./outputs", exist_ok=True)


def main():
    """Main function for training the model."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="Path to train data")
    parser.add_argument("--cv", type=int, default = 5, help="Path to train data")
    parser.add_argument("--top_models", type=str, help="Path to train data")
    parser.add_argument("--train_data", type=str, help="Path to train data")
    parser.add_argument("--test_data", type=str, help="Path to train data")
    args = parser.parse_args()
    
    num_features,cat_features = [], []
    # Load and prepare training data
    df = pd.read_csv(select_first_file(args.data))

    for col in df.columns:
        if col.endswith('_num'):
            num_features.append(col)
        if col.endswith('_cat'):
            cat_features.append(col)
            
    X = df.drop(columns = "price_range")
    y = df['price_range']

    print(f"Training with data of shape {X.shape}")

    random_state=42

    scalers = [MinMaxScaler(), StandardScaler(), RobustScaler()]
    
    svc_model = SVC(random_state = random_state) 
    gbc_model = GradientBoostingClassifier(random_state = random_state)
    rf_model = RandomForestClassifier(random_state = random_state)
    knn_model = KNeighborsClassifier()
    xgb_model = XGBClassifier(random_state = random_state)
    lgb_model = LGBMClassifier(random_state = random_state, verbose= -1)
    cat_model = CatBoostClassifier(random_state = random_state, logging_level='Silent')
    
    models = ('SVM', svc_model), ('GB', gbc_model), ('RF', rf_model), ('KNN', knn_model), ('XGB', xgb_model), ('LGB', lgb_model), ('CAT', cat_model)

    results = []

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for scaler in scalers:
        for model_name, model in models:
            ave_precision, ave_recall, ave_f1, ave_acc = evaluate_model(scaler, X_train, y_train, num_features, cat_features, model, args.cv)
    
            results.append({
                'Scaler': scaler,
                'Model': model_name,
                'Precision': ave_precision,
                'Recall': ave_recall,
                'F1 Score': ave_f1,
                'Accuracy': ave_acc
                
            })

    df_model = pd.DataFrame(results).sort_values(by='F1 Score', ascending=False).reset_index(drop=True)
    print(df_model)

    df_model_sorted = df_model.sort_values(by='F1 Score', ascending=False)
    top_unique_models = df_model_sorted.drop_duplicates(subset='Model')
    top_models_df = top_unique_models.head(3)

    print(f"Top 3 models:\n{top_models_df}")

    top_3_csv = os.path.join(args.top_models, "top_3_models.csv")
    top_models_df.to_csv(top_3_csv, index=False)

    train_df = pd.concat([X_train.reset_index(drop=True),y_train.reset_index(drop=True)], axis = 1)
    train_data_csv = os.path.join(args.train_data, "train_data_data.csv")
    train_df.to_csv(train_data_csv, index=False)
    
    test_df = pd.concat([X_test.reset_index(drop=True),y_test.reset_index(drop=True)], axis = 1)
    test_data_csv = os.path.join(args.test_data, "test_data_data.csv")
    test_df.to_csv(test_data_csv, index=False)

    mlflow.log_artifact(top_3_csv)         
    mlflow.end_run()


if __name__ == "__main__":
    main()
