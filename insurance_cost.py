# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 10:21:00 2023

@author: Lenovo
"""

import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open('C:/Users/Lenovo/Desktop/minor/deploy/model/medical_insurance_cost.sav', 'rb'))

def insurance_cost_prediction(input_data):
    input_data = (31,1,25.74,0,1,0)

    # changing input_data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    return 'The insurance cost is USD: ', prediction[0]

def main():
    
    
    # giving a title
    st.title('Insurance Cost Prediction using Ml')
    
    
    # getting the input data from the user
    
    
    Age = st.number_input('Enter your Age')
    Sex = st.number_input('SEX(0-Male/1-Female)')
    BMI = st.number_input('Body Mass Index')
    Children = st.number_input('No of children')
    Smoker = st.number_input('Do you smoke?(0-Yes/1-No)')
    Region = st.number_input('Enter your region(0-southeast/1-southwest/2-northeast/3-northwest)')
    
    
    # code for Prediction
    diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Test Result'):
        diagnosis = insurance_cost_prediction([Age, Sex, BMI, Children, Smoker, Region])        
        
    st.success(diagnosis)
    
    
    
    
    
if __name__ == '__main__':
    main()