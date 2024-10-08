# Overview

In this project, I developed a mobile specification classification model using the Azure Machine Learning Python SDK v2, alongside Azure AutoML for comparison. The project involves building an end-to-end machine learning pipeline that includes data preprocessing, feature engineering, model training, fine-tuning, and deployment within Azure's scalable cloud environment. After deployment, I used Postman to validate the model by sending API requests to the deployed endpoint, ensuring the model provided real-time predictions, making it accessible for integration with external applications via RESTful APIs.

I created a manually optimized pipeline using the Azure ML SDK v2 and compared its performance to an Azure AutoML approach. The AutoML model achieved a weighted F1 score of 93%, while the manually optimized pipeline reached an impressive 97%. This highlights the advantage of manual feature selection and tuning in boosting classification accuracy, demonstrating the benefits of custom machine learning solutions over automated ones.

# Dataset

The Mobile Price Classification Dataset [Kaggle](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification) which contains mobile phone specifications and their corresponding price ranges. The dataset includes features such as battery power, RAM, internal memory, camera resolutions, screen size, and whether the phone supports 4G or NFC.

The target variable, Price Range, classifies phones into four categories:

0: Low
1: Medium
2: High
3: Very high

# Manually Optimized Azure ML Pipeline

## 1. Data Preprocessing
- **Handle missing values to ensure data integrity.**
- **Remove duplicate values to avoid redundant information.**
  
## 2. Feature Engineering
- **Check for skewness and apply Box-Cox transformation for large skew values.**
- **Use Chi-Square tests for categorical feature selection and select those with p-values below 0.05.**
- **Use ANOVA for numerical features and select those with p-values below 0.05.**
  
## 3. Model Selection
- **Split the data** into training and testing sets and apply 5-fold cross-validation on the training data.
- **Create a pipeline** that tests different scalers (MinMaxScaler, StandardScaler, RobustScaler) and various machine learning algorithms (SVM, Gradient Boosting, Random Forest, XGBoost, LightGBM, CatBoost).
- **Compare and select the top 3 models** based on the weighted F1 score for further fine-tuning.
  
### Top 3 Validation Results

  | Scaler             | Baseline Model | Precision | Recall   | F1 Score | Accuracy  |
|--------------------|-------|-----------|----------|----------|-----------|
| RobustScaler()     | CatBoost   | 0.937639  | 0.936875 | 0.936813 | 0.936875  |
| StandardScaler()   | SVM   | 0.922122  | 0.921250 | 0.921353 | 0.921250  |
| StandardScaler()   | LightGBM   | 0.915184  | 0.914375 | 0.914389 | 0.914375  |

## 4. Model FineTuning 
- **Tuning method: Optuna was used to fine-tune hyperparameters for each model.**
- **Best models: The table below shows the performance of top models after fine-tuning.**
  
| Scaler             | Model       | Precision | Recall   | F1 Score | Accuracy  |
|--------------------|-------------|-----------|----------|----------|-----------|
| StandardScaler()     | SVM    | 0.960625  | 0.96091 | 0.96062 | 0.96055  |
| RobustScaler()   | CatBoost         | 0.94500  | 0.945139 | 0.9450 | 0.94493  |
| StandardScaler()   | LightGBM    | 0.925  | 0.925770 | 0.925 | 0.92519  |


## 5. Evaluation 

- **Evaluate top models on the test dataset using metrics such as accuracy, F1-score, precision, and recall.**
- **Select the best model, StandardScaler with SVM, based on fine-tuning results and overall performance.**
- **Register the best model in MLFlow for tracking and deployment.**

| Scaler             | Model       | Precision | Recall   | F1 Score | 
|--------------------|-------------|-----------|----------|----------|
| StandardScaler()     | SVM    | 0.96812  | 0.9675 | 0.96743 | 
| RobustScaler()   | CatBoost         | 0.95276  | 0.9525 | 0.95229 | 
| StandardScaler()   | LightGBM    | 0.9245  | 0.9225 | 0.92274 | 

## 6. Model Deployment with Postman

After selecting the best model (SVM with StandardScaler), I deployed it to an Azure Managed Online Endpoint for real-time predictions. Postman was used to test the API by sending POST requests to the endpoint with the input data in JSON format.

Steps:
- **Configure Postman: Set up the endpoint URL, add the Authorization token, and set the content type to application/json.**
- **Input Data: Send the input features as key-value pairs in the JSON body.**
- **Send Request: Postman sends a POST request, and the model responds with predictions.**
- **This process confirmed the successful deployment and real-time inference of the model.**

<img src="https://github.com/user-attachments/assets/6395debb-f4b3-4bd6-a714-c8632182538c" width="500" />

## Azure ML Pipeline

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

| Model Type       | Metric      | Validation Score | Test Score |
|------------------|-------------|------------------|------------|
| Voting Ensemble  | Accuracy    | 0.93750          | 0.93757    |
| Voting Ensemble  | Precision   | 0.93768          | 0.9375     |
| Voting Ensemble  | Recall      | 0.93750          | 0.9375     |
| Voting Ensemble  | F1 Score    | 0.93741          | 0.937491   |


# Conclusion

This project demonstrated the effectiveness of manually optimizing an Azure ML pipeline compared to AutoML. While AutoML achieved a respectable 93% weighted F1 score, manual optimization outperformed it with 97%, highlighting the value of feature selection and model tuning in improving classification performance. 

However, a potential hybrid approach could be explored: manual feature selection combined with AutoML for model training and hyperparameter tuning. This approach leverages the strengths of both methods—manual feature selection to ensure that only the most relevant features are used, and AutoML's efficiency in evaluating and tuning a wide range of models.

By pre-selecting features, AutoML can focus on optimizing model performance without being burdened by irrelevant or redundant data, which may lead to faster training times and potentially better results.

While this hybrid approach could be ideal in some scenarios, its success depends on factors such as dataset size and complexity. It strikes a balance between automation and manual control, combining the advantages of domain knowledge and computational efficiency.
