from utils import load_data, select_first_file, objective, train_model
import argparse
import os
import pandas as pd
import mlflow
import optuna
from joblib import dump, load
from sklearn.metrics import accuracy_score, precision_score, recall_score
mlflow.start_run()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True, help="Path to train data directory")
    parser.add_argument("--top_model_3", type=str, help="Type of model to optimize")
    parser.add_argument("--n_trials", type=int, help="Type of model to optimize")
    parser.add_argument("--model_3", type=str, help="Type of model to optimize")
    args = parser.parse_args()

    train_data = select_first_file(args.train_data)
    X_train, y_train, num_features, cat_features = load_data(train_data)

    top_models_df = pd.read_csv(select_first_file(args.top_model_3))
    top_model = top_models_df.iloc[2]
    scaler = top_model['Scaler']
    model_type = top_model['Model']
   
    study = optuna.create_study(directions=['maximize', 'maximize', 'maximize', 'maximize'])
    study.optimize(lambda trial: objective(trial, scaler, model_type, X_train, y_train, num_features, cat_features, 5), n_trials=args.n_trials)
    

    best_trial = study.best_trials[0]

    best_accuracy, best_precision, best_recall, best_f1_score = best_trial.values[0], best_trial.values[1], best_trial.values[2], best_trial.values[3]
    
    print("Best Accuracy:", best_accuracy)
    print("Best Precision:", best_precision)
    print("Best Recall:", best_recall)
    print("Best F1:", best_f1_score)
    print("Best trial parameters:", best_trial.params)

    mlflow.log_metric('best_f1_score', best_f1_score)


    pipeline = train_model(model_type, scaler, X_train, y_train,num_features,  cat_features, best_trial)
    pipeline.fit(X_train, y_train)
 
    results_df = pd.DataFrame({
        'Model': [model_type],
        'Best_F1_Score': [best_f1_score],
        'Best_Accuracy': [best_accuracy],
        'Best_Precision': [best_precision],
        'Best_Recall': [best_recall],
        'Best Parameter': [best_trial.params],
        
    })

    results_path = os.path.join(args.model_3, "Model_best_3.csv")
    results_df.to_csv(results_path, index=False)
    mlflow.log_artifact(results_path) 

    model_saved_path = os.path.join(args.model_3, "trained_mode3.pkl")  
    dump(pipeline, model_saved_path)  
    mlflow.log_artifact(model_saved_path)

    mlflow.end_run()

if __name__ == "__main__":
    main()
