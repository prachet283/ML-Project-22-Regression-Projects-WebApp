# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 16:25:58 2024

@author: prachet
"""
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import json
import pandas as pd


#loading the saved model of house price prediction
with open("Preprocessing File/ML-Project-3-House_Price_Prediction/Updated/columns.pkl", 'rb') as f:
    all_columns_house_price = pickle.load(f)
with open("C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/ML-Project-3-House_Price_Prediction/Updated/scaler.pkl", 'rb') as f:
    scaler_house_price = pickle.load(f)
with open("C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/ML-Project-3-House_Price_Prediction/Updated/best_features_lr.json", 'r') as file:
    best_features_lr_house_price = json.load(file)
with open("C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/ML-Project-3-House_Price_Prediction/Updated/best_features_rfr.json", 'r') as file:
    best_features_rfr_house_price = json.load(file)
with open("C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/ML-Project-3-House_Price_Prediction/Updated/best_features_xgb.json", 'r') as file:
    best_features_xgb_house_price = json.load(file)
with open("C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/ML-Project-3-House_Price_Prediction/Updated/house_price_prediction_trained_lr_model.sav", 'rb') as f:
    loaded_model_lr_house_price = pickle.load(f)
with open("C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/ML-Project-3-House_Price_Prediction/Updated/house_price_prediction_trained_rfr_model.sav", 'rb') as f:
    loaded_model_rfr_house_price = pickle.load(f)
with open("C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/ML-Project-3-House_Price_Prediction/Updated/house_price_prediction_trained_xgb_model.sav", 'rb') as f:
    loaded_model_xgb_house_price = pickle.load(f)



#loading the saved model of car price prediction
with open("C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/ML-Project-7-Car Price Prediction/Updated/columns.pkl", 'rb') as f:
    all_columns_car_price = pickle.load(f)
with open("C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/ML-Project-7-Car Price Prediction/Updated/cat_columns.pkl", 'rb') as f:
    cat_columns_car_price = pickle.load(f)
with open("C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/ML-Project-7-Car Price Prediction/Updated/encoder.pkl", 'rb') as f:
    encoder_car_price = pickle.load(f)
with open("C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/ML-Project-7-Car Price Prediction/Updated/encoded_columns.pkl", 'rb') as f:
    encoded_columns_car_price = pickle.load(f)
with open("C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/ML-Project-7-Car Price Prediction/Updated/training_columns.pkl", 'rb') as f:
    training_columns_car_price = pickle.load(f)
with open("C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/ML-Project-7-Car Price Prediction/Updated/scaler.pkl", 'rb') as f:
    scaler_car_price = pickle.load(f)
with open("C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/ML-Project-7-Car Price Prediction/Updated/best_features_xgb.json", 'r') as file:
    best_features_xgb_car_price = json.load(file)
with open("C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/ML-Project-7-Car Price Prediction/Updated/best_features_rfr.json", 'r') as file:
    best_features_rfr_car_price = json.load(file)
with open("C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/ML-Project-7-Car Price Prediction/Updated/best_features_lr.json", 'r') as file:
    best_features_lr_car_price = json.load(file)
with open("C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/ML-Project-7-Car Price Prediction/Updated/car_price_trained_xgb_model.sav", 'rb') as f:
    loaded_model_xgb_car_price = pickle.load(f)
with open("C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/ML-Project-7-Car Price Prediction/Updated/car_price_trained_rfr_model.sav", 'rb') as f:
    loaded_model_rfr_car_price = pickle.load(f)
with open("C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/ML-Project-7-Car Price Prediction/Updated/car_price_trained_lr_model.sav", 'rb') as f:
    loaded_model_lr_car_price = pickle.load(f)


#loading the saved model of gold price prediction
with open("C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/ML-Project-8-Gold Price Prediction/Updated/columns.pkl", 'rb') as f:
    all_columns_gold_price = pickle.load(f)
with open("C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/ML-Project-8-Gold Price Prediction/Updated/scaler.pkl", 'rb') as f:
    scalers_gold_price = pickle.load(f)
with open("C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/ML-Project-8-Gold Price Prediction/Updated/best_features_knr.json", 'r') as file:
    best_features_knr_gold_price = json.load(file)
with open("C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/ML-Project-8-Gold Price Prediction/Updated/best_features_rfr.json", 'r') as file:
    best_features_rfr_gold_price = json.load(file)
with open("C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/ML-Project-8-Gold Price Prediction/Updated/best_features_xgb.json", 'r') as file:
    best_features_xgb_gold_price = json.load(file)
with open("C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/ML-Project-8-Gold Price Prediction/Updated/gold_price_prediction_trained_knr_model.sav", 'rb') as f:
    loaded_model_knr_gold_price = pickle.load(f)
with open("C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/ML-Project-8-Gold Price Prediction/Updated/gold_price_prediction_trained_rfr_model.sav", 'rb') as f:
    loaded_model_rfr_gold_price = pickle.load(f)
with open("C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/ML-Project-8-Gold Price Prediction/Updated/gold_price_prediction_trained_xgb_model.sav", 'rb') as f:
    loaded_model_xgb_gold_price = pickle.load(f)


#loading the saved model of medical insurance cost prediction
with open("C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/ML-Project-11-Medical Insurance Cost Prediction/Updated/columns.pkl", 'rb') as f:
    all_columns_medical_insurance_cost = pickle.load(f)
with open("C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/ML-Project-11-Medical Insurance Cost Prediction/Updated/cat_columns.pkl", 'rb') as f:
    cat_columns_medical_insurance_cost = pickle.load(f)
with open("C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/ML-Project-11-Medical Insurance Cost Prediction/Updated/encoder.pkl", 'rb') as f:
    encoder_medical_insurance_cost = pickle.load(f)
with open("C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/ML-Project-11-Medical Insurance Cost Prediction/Updated/encoded_columns.pkl", 'rb') as f:
    encoded_columns_medical_insurance_cost = pickle.load(f)
with open("C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/ML-Project-11-Medical Insurance Cost Prediction/Updated/training_columns.pkl", 'rb') as f:
    training_columns_medical_insurance_cost = pickle.load(f)
with open("C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/ML-Project-11-Medical Insurance Cost Prediction/Updated/scaler.pkl", 'rb') as f:
    scaler_medical_insurance_cost = pickle.load(f)
with open("C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/ML-Project-11-Medical Insurance Cost Prediction/Updated/best_features_xgb.json", 'r') as file:
    best_features_xgb_medical_insurance_cost = json.load(file)
with open("C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/ML-Project-11-Medical Insurance Cost Prediction/Updated/best_features_rfr.json", 'r') as file:
    best_features_rfr_medical_insurance_cost = json.load(file)
with open("C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/ML-Project-11-Medical Insurance Cost Prediction/Updated/best_features_knr.json", 'r') as file:
    best_features_knr_medical_insurance_cost = json.load(file)
with open("C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/ML-Project-11-Medical Insurance Cost Prediction/Updated/medical_insurance_price_trained_rfr_model.sav", 'rb') as f:
    loaded_model_rfr_medical_insurance_cost = pickle.load(f)
with open("C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/ML-Project-11-Medical Insurance Cost Prediction/Updated/medical_insurance_price_trained_xgb_model.sav", 'rb') as f:
    loaded_model_xgb_medical_insurance_cost = pickle.load(f)
with open("C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/ML-Project-11-Medical Insurance Cost Prediction/Updated/medical_insurance_price_trained_knr_model.sav", 'rb') as f:
    loaded_model_knr_medical_insurance_cost = pickle.load(f)


#loading the saved model of big mart sales prediction
with open("C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/ML-Project-12-Big Mart Sales Prediction/Updated/columns.pkl", 'rb') as f:
    all_columns_big_mart_sales = pickle.load(f)
with open("C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/ML-Project-12-Big Mart Sales Prediction/Updated/cat_columns.pkl", 'rb') as f:
    cat_columns_big_mart_sales = pickle.load(f)
with open("C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/ML-Project-12-Big Mart Sales Prediction/Updated/encoder.pkl", 'rb') as f:
    encoder_big_mart_sales = pickle.load(f)
with open("C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/ML-Project-12-Big Mart Sales Prediction/Updated/encoded_columns.pkl", 'rb') as f:
    encoded_columns_big_mart_sales = pickle.load(f)
with open("C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/ML-Project-12-Big Mart Sales Prediction/Updated/training_columns.pkl", 'rb') as f:
    training_columns_big_mart_sales = pickle.load(f)
with open("C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/ML-Project-12-Big Mart Sales Prediction/Updated/scaler.pkl", 'rb') as f:
    scaler_big_mart_sales = pickle.load(f)
with open("C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/ML-Project-12-Big Mart Sales Prediction/Updated/best_features_ls.json", 'r') as file:
    best_features_ls_big_mart_sales = json.load(file)
with open("C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/ML-Project-12-Big Mart Sales Prediction/Updated/best_features_lr.json", 'r') as file:
    best_features_lr_big_mart_sales = json.load(file)
with open("C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/ML-Project-12-Big Mart Sales Prediction/Updated/best_features_xgb.json", 'r') as file:
    best_features_xgb_big_mart_sales = json.load(file)
with open("C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/ML-Project-12-Big Mart Sales Prediction/Updated/big_mart_sales_trained_ls_model.sav", 'rb') as f:
    loaded_model_ls_big_mart_sales = pickle.load(f)
with open("C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/ML-Project-12-Big Mart Sales Prediction/Updated/big_mart_sales_trained_lr_model.sav", 'rb') as f:
    loaded_model_lr_big_mart_sales = pickle.load(f)
with open("C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/ML-Project-12-Big Mart Sales Prediction/Updated/big_mart_sales_trained_xgb_model.sav", 'rb') as f:
    loaded_model_xgb_big_mart_sales = pickle.load(f)


#loading the saved model calorie burnt prediction
with open("C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/ML-Project-16-Calories Burnt Prediction using Machine Learning/Updated/columns.pkl", 'rb') as f:
    all_columns_calorie_burnt = pickle.load(f)
with open("C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/ML-Project-16-Calories Burnt Prediction using Machine Learning/Updated/scaler.pkl", 'rb') as f:
    scalers_calorie_burnt = pickle.load(f)
with open("C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/ML-Project-16-Calories Burnt Prediction using Machine Learning/Updated/best_features_xgb.json", 'r') as file:
    best_features_xgb_calorie_burnt = json.load(file)
with open("C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/ML-Project-16-Calories Burnt Prediction using Machine Learning/Updated/best_features_rfr.json", 'r') as file:
    best_features_rfr_calorie_burnt = json.load(file)
with open("C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/ML-Project-16-Calories Burnt Prediction using Machine Learning/Updated/best_features_knr.json", 'r') as file:
    best_features_knr_calorie_burnt = json.load(file)
with open("C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/ML-Project-16-Calories Burnt Prediction using Machine Learning/Updated/calories_burnt_prediction_trained_xgb_model.sav", 'rb') as f:
    loaded_model_xgb_calorie_burnt = pickle.load(f)
with open("C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/ML-Project-16-Calories Burnt Prediction using Machine Learning/Updated/calories_burnt_prediction_trained_rfr_model.sav", 'rb') as f:
    loaded_model_rfr_calorie_burnt = pickle.load(f)
with open("C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/ML-Project-16-Calories Burnt Prediction using Machine Learning/Updated/calories_burnt_prediction_trained_knr_model.sav", 'rb') as f:
    loaded_model_knr_calorie_burnt = pickle.load(f)

#creating a function for house price prediction
def house_price_prediction(input_data):

    # Convert the tuple to a DataFrame
    df = pd.DataFrame([input_data], columns=all_columns_house_price)
    
    #standardizing the data
    df[all_columns_house_price] = scaler_house_price.transform(df[all_columns_house_price])
    
    #loading best features
    df_best_features_lr_house_price = df[best_features_lr_house_price]
    df_best_features_rfr_house_price = df[best_features_rfr_house_price]
    df_best_features_xgb_house_price = df[best_features_xgb_house_price]
    
    #predictions
    prediction1_house_price = loaded_model_lr_house_price.predict(df_best_features_lr_house_price)
    prediction2_house_price = loaded_model_rfr_house_price.predict(df_best_features_rfr_house_price)
    prediction3_house_price = loaded_model_xgb_house_price.predict(df_best_features_xgb_house_price)
    
    return prediction1_house_price , prediction2_house_price, prediction3_house_price



#creating a function for car price prediction
def car_price_prediction(input_data):

    # Convert the tuple to a DataFrame
    df = pd.DataFrame([input_data], columns=all_columns_car_price)
    
    # Encode the categorical columns
    input_data_encoded = encoder_car_price.transform(df[cat_columns_car_price])
    
    # Create a DataFrame with the encoded features
    input_data_encoded_df = pd.DataFrame(input_data_encoded, columns=encoded_columns_car_price)
    
    # Add the remaining non-categorical columns
    input_data_final_encoded = pd.concat([df.drop(cat_columns_car_price, axis=1).reset_index(drop=True), input_data_encoded_df], axis=1)
    
    # Standardize the input data
    input_data_scaled = scaler_car_price.transform(input_data_final_encoded)
    
    # Create a DataFrame with the standardized features
    input_data_df = pd.DataFrame(input_data_scaled, columns=training_columns_car_price)
    
    #loading best features
    df_best_features_xgb_car_price = input_data_df[best_features_xgb_car_price]
    df_best_features_rfr_car_price = input_data_df[best_features_rfr_car_price]
    df_best_features_lr_car_price = input_data_df[best_features_lr_car_price]
    
    #predictions
    prediction1_car_price = loaded_model_xgb_car_price.predict(df_best_features_xgb_car_price)
    prediction2_car_price = loaded_model_rfr_car_price.predict(df_best_features_rfr_car_price)
    prediction3_car_price = loaded_model_lr_car_price.predict(df_best_features_lr_car_price)
    
    return prediction1_car_price , prediction2_car_price , prediction3_car_price


#creating a function for gold price prediction
def gold_price_prediction(input_data):

    # Convert the tuple to a DataFrame
    df = pd.DataFrame([input_data], columns=all_columns_gold_price)
    
    #loading columns
    df[all_columns_gold_price] = scalers_gold_price.transform(df[all_columns_gold_price])
    
    #loading best features
    df_best_features_knr_gold_price = df[best_features_knr_gold_price]
    df_best_features_rfr_gold_price = df[best_features_rfr_gold_price]
    df_best_features_xgb_gold_price = df[best_features_xgb_gold_price]
    
    #predictions
    prediction1_gold_price = loaded_model_knr_gold_price.predict(df_best_features_knr_gold_price)
    prediction2_gold_price = loaded_model_rfr_gold_price.predict(df_best_features_rfr_gold_price)
    prediction3_gold_price = loaded_model_xgb_gold_price.predict(df_best_features_xgb_gold_price)
    
    return prediction1_gold_price , prediction2_gold_price , prediction3_gold_price


#creating a function for medical insurance cost prediction
def medical_insurance_cost_prediction(input_data):

    # Convert the tuple to a DataFrame
    df = pd.DataFrame([input_data], columns=all_columns_medical_insurance_cost)
    
    # Encode the categorical columns
    input_data_encoded = encoder_medical_insurance_cost.transform(df[cat_columns_medical_insurance_cost])
    
    # Create a DataFrame with the encoded features
    input_data_encoded_df = pd.DataFrame(input_data_encoded, columns=encoded_columns_medical_insurance_cost)
    
    # Add the remaining non-categorical columns
    input_data_final_encoded = pd.concat([df.drop(cat_columns_medical_insurance_cost, axis=1).reset_index(drop=True), input_data_encoded_df], axis=1)
    
    # Standardize the input data
    input_data_scaled = scaler_medical_insurance_cost.transform(input_data_final_encoded)
    
    # Create a DataFrame with the standardized features
    input_data_df = pd.DataFrame(input_data_scaled, columns=training_columns_medical_insurance_cost)
    
    #loading best features
    df_best_features_rfr_medical_insurance_cost = input_data_df[best_features_rfr_medical_insurance_cost]
    df_best_features_xgb_medical_insurance_cost = input_data_df[best_features_xgb_medical_insurance_cost]
    df_best_features_knr_medical_insurance_cost = input_data_df[best_features_knr_medical_insurance_cost]
    
    #predictions
    prediction1_medical_insurance_cost = loaded_model_rfr_medical_insurance_cost.predict(df_best_features_rfr_medical_insurance_cost)
    prediction2_medical_insurance_cost = loaded_model_xgb_medical_insurance_cost.predict(df_best_features_xgb_medical_insurance_cost)
    prediction3_medical_insurance_cost = loaded_model_knr_medical_insurance_cost.predict(df_best_features_knr_medical_insurance_cost)

    return prediction1_medical_insurance_cost , prediction2_medical_insurance_cost , prediction3_medical_insurance_cost


#creating a function for big mart sales prediction
def big_mart_sales_prediction(input_data):
    
    # Convert the tuple to a DataFrame
    df = pd.DataFrame([input_data], columns=all_columns_big_mart_sales)
    
    # Encode the categorical columns
    input_data_encoded = encoder_big_mart_sales.transform(df[cat_columns_big_mart_sales])
    
    # Create a DataFrame with the encoded features
    input_data_encoded_df = pd.DataFrame(input_data_encoded, columns=encoded_columns_big_mart_sales)
    
    # Add the remaining non-categorical columns
    input_data_final_encoded = pd.concat([df.drop(cat_columns_big_mart_sales, axis=1).reset_index(drop=True), input_data_encoded_df], axis=1)
    
    # Standardize the input data
    input_data_scaled = scaler_big_mart_sales.transform(input_data_final_encoded)
    
    # Create a DataFrame with the standardized features
    input_data_df = pd.DataFrame(input_data_scaled, columns=training_columns_big_mart_sales)
    
    #loading best features
    df_best_features_ls_big_mart_sales = input_data_df[best_features_ls_big_mart_sales]
    df_best_features_lr_big_mart_sales = input_data_df[best_features_lr_big_mart_sales]
    df_best_features_xgb_big_mart_sales = input_data_df[best_features_xgb_big_mart_sales]
    
    #predictions
    prediction1_big_mart_sales = loaded_model_ls_big_mart_sales.predict(df_best_features_ls_big_mart_sales)
    prediction2_big_mart_sales = loaded_model_lr_big_mart_sales.predict(df_best_features_lr_big_mart_sales)
    prediction3_big_mart_sales = loaded_model_xgb_big_mart_sales.predict(df_best_features_xgb_big_mart_sales)

    return prediction1_big_mart_sales , prediction2_big_mart_sales , prediction3_big_mart_sales


#creating a function for calorie burnt prediction
def calorie_burnt_prediction(input_data):

    # Convert the tuple to a DataFrame
    df = pd.DataFrame([input_data], columns=all_columns_calorie_burnt)
    
    #loading columns
    df[all_columns_calorie_burnt] = scalers_calorie_burnt.transform(df[all_columns_calorie_burnt])
    
    #loading best features
    df_best_features_knr_calorie_burnt = df[best_features_knr_calorie_burnt]
    df_best_features_rfr_calorie_burnt = df[best_features_rfr_calorie_burnt]
    df_best_features_xgb_calorie_burnt = df[best_features_xgb_calorie_burnt]
    
    #predictions
    prediction1_calorie_burnt = loaded_model_xgb_calorie_burnt.predict(df_best_features_xgb_calorie_burnt)
    prediction2_calorie_burnt = loaded_model_rfr_calorie_burnt.predict(df_best_features_rfr_calorie_burnt)
    prediction3_calorie_burnt = loaded_model_knr_calorie_burnt.predict(df_best_features_knr_calorie_burnt)

    return prediction1_calorie_burnt , prediction2_calorie_burnt , prediction3_calorie_burnt


def main():
    # sidebar for navigate

    with st.sidebar:
    
        selected = option_menu('ML Regression Projects WebApp System',
                           
                            ['House Price Prediction',
                            'Car Price Prediction',
                            'Gold Price Prediction',
                            'Medical Insurance Cost Prediction',
                            'Big Mart Sales Prediction',
                            'Calorie Burnt Prediction'],
                           
                           icons = ['house-door','car-front-fill','gem','heart-pulse','cart','fire'],
                           
                           default_index = 0)
        
    

    # House Price Prediction Page
    if( selected == 'House Price Prediction'):
        
        #giving a title
        st.title('House Price Prediction Web App')
        
        col1 , col2 , col3 = st.columns(3)
        #getting input data from user
        with col1:
            SquareMeters = st.number_input("Size of house in square meters",format="%.2f")
        with col2:
            NumberOfRooms = st.number_input("Number Of Rooms",format="%.0f")
        with col3:
            option1 = st.selectbox('Has Yard',('No', 'Yes')) 
            HasYard = 1 if option1 == 'Yes' else 0
        with col1:
            option2 = st.selectbox('Has Pool',('No', 'Yes')) 
            HasPool = 1 if option2 == 'Yes' else 0
        with col2:
            Floors = st.number_input("Number of floors",format="%.0f")
        with col3:
            CityCode = st.number_input("City Code",format="%.0f")
        with col1:
            CityPartRange = st.selectbox('City Part Range(cheapest to expensive)',('1','2','3','4','5','6','7','8','9','10')) 
        with col2:
            NumPrevOwners = st.number_input("Num Prev Owners",format="%.0f")
        with col3:
            Made = st.number_input("Made in Year",format="%.0f")
        with col1:
            option3 = st.selectbox('Is New Built',('No', 'Yes'))
            IsNewBuilt = 1 if option3 == 'Yes' else 0
        with col2:
            option4 = st.selectbox('Has Storm Protector',('No', 'Yes'))
            HasStormProtector = 1 if option4 == 'Yes' else 0
        with col3:
            Basement = st.number_input('Basement in square meters',format="%.2f")
        with col1:
            Attic = st.number_input('Attic in square meteres',format="%.2f")
        with col2:
            Garage = st.number_input('Garage Size in square meteres',format="%.2f")
        with col3:
            option5 = st.selectbox('Has Storage Room',('No', 'Yes'))
            HasStorageRoom = 1 if option5 == 'Yes' else 0
        with col1:
            HasGuestRoom = st.number_input('Number of guest rooms',format="%.0f")	
        
        
        # code for prediction
        house_price_prediction_lr = ''
        house_price_prediction_rfr = ''
        house_price_prediction_xgb = ''
        
    
        house_price_prediction_lr,house_price_prediction_rfr,house_price_prediction_xgb = house_price_prediction([SquareMeters,NumberOfRooms,HasYard,HasPool,Floors,CityCode,CityPartRange,NumPrevOwners,Made,IsNewBuilt,HasStormProtector,Basement,Attic,Garage,HasStorageRoom,HasGuestRoom])
            
        #creating a button for Prediction
        if st.button("Predict House Price"):
            prediction = house_price_prediction_lr[0]
            prediction = "{:.2f}".format(prediction)
            st.write(f"The Predicted Price: {prediction} $")
        
        if st.checkbox("Show Advanced Options"):
            if st.button("Predict House Price with Linear Regression Model"):
                prediction = house_price_prediction_lr[0]
                prediction = "{:.2f}".format(prediction)
                st.write(f"The Predicted Price: {prediction} $")
            if st.button("Predict House Price with Random Forest Regressor Model"):
                prediction = house_price_prediction_rfr[0]
                prediction = "{:.2f}".format(prediction)
                st.write(f"The Predicted Price: {prediction} $")
            if st.button("Predict House Price with XG Boost Regressor"):
                prediction = house_price_prediction_xgb[0]
                prediction = "{:.2f}".format(prediction)
                st.write(f"The Predicted Price: {prediction} $")  
    
 
    # Car Price Prediction Page
    if( selected == 'Car Price Prediction'):
        
        #giving a title
        st.title('Car Price Prediction Web App')
        
        col1 , col2 = st.columns(2)
        #getting input data from user
        with col1:
            year = st.number_input("Year",format="%.0f")
        with col2:
            km_driven = st.number_input("km_driven",format="%.2f")
        with col1:
            fuel = st.selectbox('fuel',('Diesel', 'Petrol','CNG')) 
        with col2:	
            seller_type = st.selectbox('seller_type',('Individual', 'Dealer')) 
        with col1:
            transmission = st.selectbox('transmission',('Manual', 'Automatic')) 
        with col2:
            owner = st.selectbox('owner',('First Owner', 'Second Owner','Third Owner','Fourth & Above Owner')) 
        
        # code for prediction
        car_price_prediction_xgb = ''
        car_price_prediction_rfr = ''
        car_price_prediction_lr = ''
        
    
        car_price_prediction_xgb,car_price_prediction_rfr,car_price_prediction_lr = car_price_prediction((year,km_driven,fuel,seller_type,transmission,owner))
            
        #creating a button for Prediction
        if st.button("Predict Car Price"):
            prediction = car_price_prediction_rfr[0]
            prediction = "{:.2f}".format(prediction)
            st.write(f"The Predicted Price: {prediction} $")
        
        if st.checkbox("Show Advanced Options"):
            if st.button("Predict Car Price with Random Forest Regressor Model"):
                prediction = car_price_prediction_rfr[0]
                prediction = "{:.2f}".format(prediction)
                st.write(f"The Predicted Price: {prediction} $")
            if st.button("Predict Car Price with XG Boost Regressor Model"):
                prediction = car_price_prediction_xgb[0]
                prediction = "{:.2f}".format(prediction)
                st.write(f"The Predicted Price: {prediction} $")
            if st.button("Predict Car Price with Linear Regression Model"):
                prediction = car_price_prediction_lr[0]
                prediction = "{:.2f}".format(prediction)
                st.write(f"The Predicted Price: {prediction} $") 

    # Gold Price Prediction Page
    if( selected == 'Gold Price Prediction'):
        
        #giving a title
        st.title('Gold Price Prediction Web App')
    
        #getting input data from user
        SPX = st.number_input("SPX",format="%.6f")
        USO = st.number_input("USO",format="%.6f")
        SLV = st.number_input("SLV",format="%.6f")	
        EUR_USD = st.number_input("EUR/USD",format="%.6f")
    
        # code for prediction
        gold_price_prediction_knr = ''
        gold_price_prediction_rfr = ''
        gold_price_prediction_xgb = ''
        
    
        gold_price_prediction_knr,gold_price_prediction_rfr,gold_price_prediction_xgb = gold_price_prediction((SPX,USO,SLV,EUR_USD))
            
        #creating a button for Prediction
        if st.button("Predict Gold Price"):
            prediction = gold_price_prediction_rfr[0]
            prediction = "{:.4f}".format(prediction)
            st.write(f"The Predicted Price: {prediction} $")
        
        if st.checkbox("Show Advanced Options"):
            if st.button("Predict Gold Price with K Neighbors Regressor Model"):
                prediction = gold_price_prediction_knr[0]
                prediction = "{:.4f}".format(prediction)
                st.write(f"The Predicted Price: {prediction} $")
            if st.button("Predict Gold Price with Random Forest Regressor Model"):
                prediction = gold_price_prediction_rfr[0]
                prediction = "{:.4f}".format(prediction)
                st.write(f"The Predicted Price: {prediction} $")
            if st.button("Predict Gold Price with XG Boost Regressor Model"):
                prediction = gold_price_prediction_xgb[0]
                prediction = "{:.4f}".format(prediction)
                st.write(f"The Predicted Price: {prediction} $") 
        
    
    # Medical Insurance Cost Prediction Page
    if( selected == 'Medical Insurance Cost Prediction'):
        
        #giving a title
        st.title('Medical Insurance Cost Prediction Web App')
        
        #getting input data from user
        
        age = st.number_input("Age of the person",format="%.0f")
        sex = st.selectbox('Gender',('male', 'female')) 
        bmi = st.number_input("BMI",format="%.3f")
        children = st.number_input("No. of children",format="%.0f")
        smoker = st.selectbox('Smoker',('yes', 'no'))
        region = st.selectbox('Region',('southeast', 'southwest','northwest','northeast'))
    
    
        # code for prediction
        medical_insurance_cost_prediction_rfr = ''
        medical_insurance_cost_prediction_xgb = ''
        medical_insurance_cost_prediction_knr = ''
        
    
        medical_insurance_cost_prediction_rfr,medical_insurance_cost_prediction_xgb,medical_insurance_cost_prediction_knr = medical_insurance_cost_prediction((age,sex,bmi,children,smoker,region))
            
        #creating a button for Prediction
        if st.button("Predict Medical Insurance Cost"):
            prediction = medical_insurance_cost_prediction_rfr[0]
            prediction = "{:.4f}".format(prediction)
            st.write(f"The Predicted Price: {prediction} $")
        
        if st.checkbox("Show Advanced Options"):
            if st.button("Predict Medical Insurance Cost with Random Forest Regressor Model"):
                prediction = medical_insurance_cost_prediction_rfr[0]
                prediction = "{:.4f}".format(prediction)
                st.write(f"The Predicted Price: {prediction} $")
            if st.button("Predict Medical Insurance Cost with XG Boost Regressor Model"):
                prediction = medical_insurance_cost_prediction_xgb[0]
                prediction = "{:.4f}".format(prediction)
                st.write(f"The Predicted Price: {prediction} $")
            if st.button("Predict Medical Insurance Cost with K Neighbors Regressor Model"):
                prediction = medical_insurance_cost_prediction_knr[0]
                prediction = "{:.4f}".format(prediction)
                st.write(f"The Predicted Price: {prediction} $") 

     # Big Mart Sales Prediction Page
    if( selected == 'Big Mart Sales Prediction'):
         
        #giving a title
        st.title('Big Mart Sales Prediction Web App')
        
        col1 , col2 = st.columns(2)
        #getting input data from user
        with col1:
            Item_Weight = st.number_input("Item Weight",format="%.4f")
        with col2:
            Item_Fat_Content = st.selectbox('Item Fat Content',('Low Fat', 'Regular'))
        with col1:
            Item_Visibility	 = st.number_input("Item Visibility",format="%.4f")
        with col2:
            Item_Type = st.selectbox("Item Type",('Fruits and Vegetables','Snack Foods','Household','Frozen Foods','Dairy','Canned','Baking Goods','Health and Hygiene','Soft Drinks','Meat','Breads','Hard Drinks','Starchy Foods','Breakfast','Seafood','Others'))
        with col1:
            Item_MRP = st.number_input("Item MRP",format="%.4f")
        with col2:
            Outlet_Establishment_Year = int(st.selectbox("Outlet Establishment Year",('1985','1987','1997','1998','1999','2002','2004','2007','2009')))
        with col1:
            Outlet_Location_Type = st.selectbox("Outlet_Location_Type",('Tier 1', 'Tier 2','Tier 3'))
        with col2:
            Outlet_Type = st.selectbox("Outlet_Type",('Grocery Store', 'Supermarket Type1','Supermarket Type2','Supermarket Type3'))
        
        
        # code for prediction
        big_mart_sales_prediction_ls = ''
        big_mart_sales_prediction_lr = ''
        big_mart_sales_prediction_xgb = ''
        
    
        big_mart_sales_prediction_ls,big_mart_sales_prediction_lr,big_mart_sales_prediction_xgb = big_mart_sales_prediction((Item_Weight,Item_Fat_Content,Item_Visibility,Item_Type,Item_MRP,Outlet_Establishment_Year,Outlet_Location_Type,Outlet_Type))
            
        #creating a button for Prediction
        if st.button("Predict Big Mart Sales"):
            prediction = big_mart_sales_prediction_xgb[0]
            prediction = "{:.4f}".format(prediction)
            st.write(f"The Predicted Sales: {prediction} $")
        
        if st.checkbox("Show Advanced Options"):
            if st.button("Predict Big Mart Sales with Lasso Model"):
                prediction = big_mart_sales_prediction_ls[0]
                prediction = "{:.4f}".format(prediction)
                st.write(f"The Predicted Price: {prediction} $")
            if st.button("Predict Big Mart Sales with Linear Regression Model"):
                prediction = big_mart_sales_prediction_lr[0]
                prediction = "{:.4f}".format(prediction)
                st.write(f"The Predicted Price: {prediction} $")
            if st.button("Predict Big Mart Sales with XG Boost Regressor Model"):
                prediction = big_mart_sales_prediction_xgb[0]
                prediction = "{:.4f}".format(prediction)
                st.write(f"The Predicted Price: {prediction} $")
                
                

     # Calorie Burnt Prediction Page
    if( selected == 'Calorie Burnt Prediction'):
         
        #giving a title
        st.title('Calorie Burnt Prediction Web App')
        
        option = st.selectbox('Gender',('Male', 'Female'))
        gender = 1 if option == 'Female' else 0
        age = st.number_input("Age of the person",format="%.0f")
        height = st.number_input("Height of the person",format="%.2f")
        weight = st.number_input("Weight of the person",format="%.2f")
        duration = st.number_input("Duration of Exercise in mins",format="%.0f")
        heart_rate = st.number_input("Heart_Rate of the person",format="%.2f")
        body_temp = st.number_input("Body Temperature of the person",format="%.2f")
        
    
        # code for prediction
        calorie_burnt_prediction_xgb = ''
        calorie_burnt_prediction_rfr = ''
        calorie_burnt_prediction_knr = ''
        
    
        calorie_burnt_prediction_xgb,calorie_burnt_prediction_rfr,calorie_burnt_prediction_knr = calorie_burnt_prediction((gender,age,height,weight,duration,heart_rate,body_temp))
            
        #creating a button for Prediction
        if st.button("Predict Calories Burnt"):
            prediction = calorie_burnt_prediction_xgb[0]
            prediction = "{:.4f}".format(prediction)
            st.write(f"The Predicted Sales: {prediction} $")
        
        if st.checkbox("Show Advanced Options"):
            if st.button("Predict Calories Burnt with XG Boost Regressor Model"):
                prediction = calorie_burnt_prediction_xgb[0]
                prediction = "{:.4f}".format(prediction)
                st.write(f"The Predicted Price: {prediction} $")
            if st.button("Predict Calories Burnt with Random Forest Regressor Model"):
                prediction = calorie_burnt_prediction_rfr[0]
                prediction = "{:.4f}".format(prediction)
                st.write(f"The Predicted Price: {prediction} $")
            if st.button("Predict Calories Burnt with K Neighbors Regressor Model"):
                prediction = calorie_burnt_prediction_knr[0]
                prediction = "{:.4f}".format(prediction)
                st.write(f"The Predicted Price: {prediction} $")
    
if __name__ == '__main__':
    main()





