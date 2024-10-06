# Overview

In this project, I developed a mobile specification classification model using the Azure Machine Learning Python SDK v2, complemented by Azure AutoML for comparative analysis. The project involved creating an end-to-end machine learning pipeline that included data preprocessing, feature engineering, model training, performance fine-tuning, and deployment—all within Azure’s scalable cloud environment.

I utilized Azure ML SDK v2 to build a manually optimized pipeline and compared its performance against an AutoML approach. The AutoML method achieved an weighted F1 score of 90%, while the manually optimized pipeline reached an impressive weighted F1 score of 97%. This comparison underscores the effectiveness of manual feature selection and model tuning in enhancing classification accuracy, demonstrating the advantages of tailored machine learning solutions over automated ones.

# Manually Optimized Azure ML Pipeline

## 1. Data Preprocessing
- **Handled missing values:**
- **Remove duplicate values:**
  
## 2. Feature Engineering
- **Check for skewness and use Box-Cox if skew value is large.**
- **Apply Chi-Square test for categorical features.**
- **Apply ANOVA for numerical features.**
- **Select features with p-values lower than 0.05.**
  
## 3. Model Selection
- **Split the data** into training and testing sets and apply 5-fold cross-validation on the training data.
- **Create a pipeline** that tests different scalers (MinMaxScaler, StandardScaler, RobustScaler) and various machine learning algorithms (SVM, Gradient Boosting, Random Forest, XGBoost, LightGBM, CatBoost).
- **Compare and select the top 3 models** based on the weighted F1 score for further fine-tuning.
  
### Top 3 Validation Results

  | Scaler             | Model | Precision | Recall   | F1 Score | Accuracy  |
|--------------------|-------|-----------|----------|----------|-----------|
| RobustScaler()     | CatBoost   | 0.937639  | 0.936875 | 0.936813 | 0.936875  |
| StandardScaler()   | SVM   | 0.922122  | 0.921250 | 0.921353 | 0.921250  |
| StandardScaler()   | LightGBM   | 0.915184  | 0.914375 | 0.914389 | 0.914375  |

## 4. Model FineTuning 
- **Tuning method: Optuna was used to fine-tune hyperparameters for each model.**
- **Best models: The table below shows the performance of top models after fine-tuning.**
  
  | Scaler             | Model | Precision | Recall   | F1 Score | Accuracy  |
|--------------------|-------|-----------|----------|----------|-----------|
| StandardScaler()    | SVM    | 0.937639  | 0.936875 | 0.936813 | 0.936875  |
| SRobustScaler()    | CAT   | 0.922122  | 0.921250 | 0.921353 | 0.921250  |
| StandardScaler()   | LGB   | 0.915184  | 0.914375 | 0.914389 | 0.914375  |

## 5. Evaluation 

## Pipeline
<img src="https://github.com/user-attachments/assets/139ed9c1-82e1-41e6-be27-e79131fbc595" width="550" />

# Azure AutoML Implementation

## 1: Split Data

The dataset is divided into training (80%) and testing sets (20%). The training set is used for 5-fold cross-validation to ensure reliable model performance, while the testing set is reserved for final evaluation. 

## 2: AutoML Training

AutoML is trained on the training data with 5-fold cross-validation, evaluating multiple models. The parameters for AutoML include:

- **Featurization**: TabularFeaturizationSettings(mode="Auto")
- **Timeout**: 15 minutes
- **Trial Timeout**: 2 minutes
- **Max Trials**: 40
- **Enable Early Termination**: True
  
## 3: Model Evaluation

The pipeline generates two outputs: the best model and the testing dataset. The best model is selected and evaluated on the test data, with performance measured using the weighted F1 score to assess classification accuracy.

## Auto ML Pipeline

<img src="https://github.com/user-attachments/assets/bd85257f-16d1-408e-95e8-8b024308d8c5" width="450" />

## Top 5 Auto ML models
![image](https://github.com/user-attachments/assets/6e8f78ed-a45b-4bd6-8360-70f8946ca278)


## Best Model: Voting Ensemble

<img src="https://github.com/user-attachments/assets/e1962351-c77b-491f-9a36-c0e8a5f7f47d" width="200" />

## Results 

### Validation 

| Model Type         | Metric         | Score |
|--------------------|----------------|-------|
| Voting Ensemble    |    Accuracy Score    | 0.93750  |
| Voting Ensemble    | Weighted Precision Score   | 0.93768    |
| Voting Ensemble    |    Weighted Recall Score        | 0.93750    |
| Voting Ensemble    |    Weighted F1 Score        | 0.93741    |

### Testing 

| Model Type         | Metric         | Score |
|--------------------|----------------|-------|
| Voting Ensemble    |    Accuracy Score    | 0.93757   |
| Voting Ensemble    | Weighted Precision Score   | 0.9375    |
| Voting Ensemble    |    Weighted Recall Score        | 0.9375    |
| Voting Ensemble    |    Weighted F1 Score        | 0.937491   |


