
import argparse
import os
import pandas as pd
import mlflow
import numpy as np
from joblib import dump, load

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc
import optuna

def select_first_file(path):
    files = os.listdir(path)
    return os.path.join(path, files[0])
    
def load_data(data_path):
    df = pd.read_csv(data_path)
    num_features = [col for col in df.columns if col.endswith('_num')]
    cat_features = [col for col in df.columns if col.endswith('_cat')]
    
    X_train = df.drop(columns="price_range")
    y_train = df['price_range']
    return X_train, y_train, num_features, cat_features

def is_scaler(scaler):
    if scaler == 'RobustScaler()':
        scalers = RobustScaler()

    elif scaler == 'StandardScaler()':
        scalers = StandardScaler()

    elif scaler == 'MinMaxScaler()':
        scalers = MinMaxScaler()
    else:
        scalers = None
        print(f"Scaler '{scaler}' is not recognized. Returning None.")
    return scalers

def objective(trial, scaler, model_type, X_train, y_train, num_features, cat_features, cv):


    if model_type == 'CAT':
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.1)
        max_depth = trial.suggest_int('max_depth', 3, 7)
        n_estimators = trial.suggest_int('n_estimators', 2000, 3000)

        model = CatBoostClassifier(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            logging_level='Silent',
            random_state=42
        )

    elif model_type == 'SVM':
        C = trial.suggest_int('C', 1, 20)
        kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf'])
        degree = trial.suggest_int('degree', 1, 10)

        model = SVC(C=C, kernel=kernel, degree=degree, random_state=42)

    elif model_type == 'LGB':
        num_leaves = trial.suggest_int('num_leaves', 150, 500)
        learning_rate = trial.suggest_float('learning_rate', 0.05, 0.08)
        min_child_samples = trial.suggest_int('min_child_samples', 25, 50)
        n_estimators = trial.suggest_int('n_estimators', 1000, 5000)

        model = LGBMClassifier(
            num_leaves=num_leaves, 
            learning_rate=learning_rate, 
            min_child_samples=min_child_samples,
            n_estimators=n_estimators,
            random_state=42,
            verbose=-1
        )
    else:
        raise ValueError("Unsupported model type")

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', is_scaler(scaler), num_features), 
            ('cat', OneHotEncoder(), cat_features)  
        ]
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    accuracy_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='accuracy')
    precision_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='precision_weighted')
    recall_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='recall_weighted')
    f1_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1_weighted')
    
    mlflow.log_metric('average_accuracy', np.mean(accuracy_scores))
    mlflow.log_metric('average_precision', np.mean(precision_scores))
    mlflow.log_metric('average_recall', np.mean(recall_scores))
    mlflow.log_metric('average_f1_score', np.mean(f1_scores))

    return np.mean(f1_scores) , np.mean(accuracy_scores), np.mean(precision_scores), np.mean(recall_scores)

def train_model(model_type, scaler, X_train, y_train,num_features,  cat_features, best_trial):
    best_params = best_trial.params
    if model_type == 'CAT':
        model = CatBoostClassifier(**best_params)
    
    elif model_type == 'SVM':
        model = SVC(**best_params)

    elif model_type == 'LGB':
        model = LGBMClassifier(**best_params)

    else:
        raise ValueError("Unsupported model type")
        
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', is_scaler(scaler), num_features), 
            ('cat', OneHotEncoder(), cat_features)  
        ]
    )
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    return pipeline
