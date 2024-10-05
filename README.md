# Overview

In this project, I developed a mobile specification classification model using the Azure Machine Learning Python SDK v2, complemented by Azure AutoML for comparative analysis. The project involved creating an end-to-end machine learning pipeline that included data preprocessing, feature engineering, model training, performance fine-tuning, and deployment—all within Azure’s scalable cloud environment.

I utilized Azure ML SDK v2 to build a manually optimized pipeline and compared its performance against an AutoML approach. The AutoML method achieved an weighted F1 score of 90%, while the manually optimized pipeline reached an impressive weighted F1 score of 97%. This comparison underscores the effectiveness of manual feature selection and model tuning in enhancing classification accuracy, demonstrating the advantages of tailored machine learning solutions over automated ones.

# Azure AutoML Implementation

## Step 1: Split Data

The dataset is divided into training (80%) and testing sets (20%). The training set is used for 5-fold cross-validation to ensure reliable model performance, while the testing set is reserved for final evaluation. 

## Step 2: AutoML Training

AutoML is trained on the training data with 5-fold cross-validation, evaluating multiple models. The parameters for AutoML include:

- **Featurization**: TabularFeaturizationSettings(mode="Auto")
- **Timeout**: 15 minutes
- **Trial Timeout**: 2 minutes
- **Max Trials**: 40
- **Enable Early Termination**: True
  
## Step 3: Model Evaluation

The pipeline generates two outputs: the best model and the testing dataset. The best model is selected and evaluated on the test data, with performance measured using the weighted F1 score to assess classification accuracy.

## Auto ML Pipeline

<img src="https://github.com/user-attachments/assets/b8b9b6f9-05f7-4566-8df8-a8999f4e2eda" width="450" />

<img src="https://github.com/user-attachments/assets/e1962351-c77b-491f-9a36-c0e8a5f7f47d" width="400" />




![image](https://github.com/user-attachments/assets/d208d152-c918-40da-906b-1f172ae2e0c8)

# Manually Optimized Azure ML Pipeline

## Data Preprocessing

## Feature Engineering

## Model Selection

## Model FineTuning 

## Evaluation 
