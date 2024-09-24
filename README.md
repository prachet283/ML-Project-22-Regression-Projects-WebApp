# ML Regression Projects WebApp

This repository contains a web app developed using Streamlit and hosted on Streamlit Cloud. The web app integrates six different regression projects, each utilizing machine learning models to provide accurate predictions. The projects covered are:

- House Price Prediction
- Car Price Prediction
- Gold Price Prediction
- Medical Insurance Cost Prediction
- Big Mart Sales Prediction
- Calories Burnt Prediction

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Dataset Description](#dataset-description)
5. [Technologies Used](#technologies-used)
6. [Model Development Process](#model-development-process)
7. [Models Used](#models-used)
8. [Model Evaluation](#model-evaluation)
9. [Conclusion](#conclusion)
10. [Deployment](#deployment)
11. [Contributing](#contributing)
12. [Contact](#contact)

## Overview

This web application allows users to select from six different regression projects and get predictions based on the input features. Each project was developed through extensive data analysis and model selection processes, ensuring high accuracy and reliability in predictions.

## Installation

To run this project locally, please follow these steps:

1. Clone the repository
2. Navigate to the project directory
3. Install the required dependencies

```bash
git clone <repository_url>
cd <project_directory>
pip install -r requirements.txt
```

## Usage

To start the Streamlit web app, run the following command in your terminal:

```bash
streamlit run streamlit_app.py
```

This will launch the web app in your default web browser. You can select the desired regression project from the sidebar and input the required features to get a prediction.

## Dataset Description

### House Price Prediction
**Description**: Predicts house prices based on features such as location, square footage, number of bedrooms, and other property details.

### Car Price Prediction
**Description**: Predicts car prices based on attributes like brand, model, year of manufacture, mileage, and engine specifications.

### Gold Price Prediction
**Description**: Predicts the price of gold based on historical data, including currency exchange rates, inflation rates, and global financial indicators.

### Medical Insurance Cost Prediction
**Description**: Predicts insurance premiums based on features like age, BMI, smoking status, and medical conditions.

### Big Mart Sales Prediction
**Description**: Predicts sales for various items in different stores based on factors such as store type, item visibility, and marketing data.

### Calories Burnt Prediction
**Description**: Predicts the number of calories burnt based on physical activities, age, weight, and duration of exercise.

## Technologies Used
- **Programming Language**: Python
- **Web Framework**: Streamlit
- **Machine Learning Libraries**: Scikit-learn, XGBoost
- **Data Analysis and Visualization**: Pandas, NumPy, Matplotlib, Seaborn

## Model Development Process

Each classification project was developed through the following steps:

1. **Importing the Dependencies**
2. **Exploratory Data Analysis (EDA)**
3. **Data Preprocessing**
    - Handling missing values
    - Handling outliers
    - Label encoding/One-hot encoding
    - Standardizing the data
4. **Model Selection**
    - Selected the most common 5 regression models
    - Trained each model and checked cross-validation scores
    - Chose the top 3 models based on cross-validation scores
5. **Model Building and Evaluation**
    - Selected best features using Recursive Feature Elimination (RFE)
    - Performed hyperparameter tuning using Grid Search CV
    - Built the final model with the best hyperparameters and features
    - Evaluated the model using mean squared error, R-squared score, and other relevant metrics

## Models Used

The top 3 models for each classification project are as follows:

### House Price Prediction
- Linear Regression: Simple and interpretable.
- Random Forest Regressor: Effective for high-dimensional data.
- XGBoost Regressor: Known for its high performance.

### Car Price Prediction
- Linear Regression: Simple and interpretable.
- Random Forest Regressor: Effective for high-dimensional data.
- XGBoost Regressor: Known for its high performance.

### Gold Price Prediction
- K-Nearest Neighbour: Simple algorithm that works well with small datasets.
- Random Forest Regressor: Effective for high-dimensional data.
- XGBoost: Boosting algorithm known for high performance.

### Medical Insurance Cost Prediction
- K-Nearest Neighbour: Simple algorithm that works well with small datasets.
- Random Forest Regressor: Effective for high-dimensional data.
- XGBoost: Boosting algorithm known for high performance.

### Big Mart Sales Prediction
- Linear Regression: Simple and interpretable.
- Lasso Regression: Linear model with L1 regularization to handle multicollinearity and feature selection effectively.
- XGBoost: Powerful gradient boosting framework.

### Calorie Burnt Prediction
- K-Nearest Neighbour: Simple algorithm that works well with small datasets.
- Random Forest Classifier: Ensemble method that reduces overfitting.
- XGBoost: Powerful gradient boosting framework.

## Model Evaluation

### House Price Prediction Model Metrics
- **Linear Regression**: R² = 0.99999
- **Random Forest Regressor**: R² = 0.99999
- **XGBoost**: R² = 0.99998

### Car Price Prediction Model Metrics
- **Random Forest Regressor**: R² = 0.66816
- **XGBoost**: R² = 0.63702
- **Linear Regression**: R² = 0.47672

### Gold Price Prediction Model Metrics
- **XGBoost**: R² = 0.98654
- **Random Forest Regressor**: R² = 0.98880
- **K Neighbors Regressor**: R² = 0.98934

### Medical Insurance Cost Prediction Model Metrics
- **Linear Regression**: R² = 0.86308
- **XGBoost**: R² = 0.86308
- **Random Forest Regressor**: R² = 0.87258

### Big Mart Sales Prediction Model Metrics
- **XGBoost**: R² = 0.61418
- **Lasso Regression**: R² = 0.57886
- **Linear Regression**: R² = 0.57842

### Calories Burnt Prediction Model Metrics
- **Random Forest Regressor**: R² = 0.99847
- **XGBoost**: R² = 0.99935
- **K Neighbors Regressor**: R² = 0.99513

## Conclusion
This ML Regression Projects WebApp provides an easy-to-use interface for predicting various outcomes based on input features. The models used are well-validated and tuned for high accuracy. The system aims to assist in decision-making and prediction tasks across different domains.

## Deployment
The web app is hosted on Streamlit Cloud. You can access it using the following link:

[ML Regression Projects WebApp](https://ml-project-22-regression-projects-webapp-kbsj3eax6jus4jsoxugy7.streamlit.app/)

## Contributing
Contributions are welcome! If you have any suggestions or improvements, please create a pull request or open an issue.

## Contact
If you have any questions or suggestions, feel free to contact me at prachetpandav283@gmail.com.

