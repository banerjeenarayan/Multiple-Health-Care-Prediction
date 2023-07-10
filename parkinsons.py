# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 22:55:54 2023

@author: Lenovo
"""

import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open('C:/Users/Lenovo/Desktop/deploy/parkinsons_model.sav', 'rb'))

# creating a function for Prediction

def parkinsons_prediction(input_data):
    

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return 'The person does not have Parkinsons Disease'
    else:
      return 'The person have Parkinsons'

def main():
    
    
    # giving a title
    st.title('Parkinsons Prediction using ML')
    
    
    # getting the input data from the user
    
    MDVP_Fo_Hz = st.text_input('MDVP:Fo(Hz)')
    MDVP_Fhi_Hz = st.text_input('MDVP:Fhi(Hz)')
    MDVP_Flo_Hz = st.text_input('MDVP:Flo(Hz)')
    MDVP_Jitter_percentage = st.text_input('MDVP:Jitter(%)')
    MDVP_Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
    MDVP_RAP = st.text_input('MDVP:RAP')
    MDVP_PPQ = st.text_input('MDVP:PPQ')
    Jitter_DDP = st.text_input('Jitter:DDP')
    MDVP_Shimmer = st.text_input('MDVP:Shimmer')
    MDVP_Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')
    Shimmer_APQ3 = st.text_input('Shimmer:APQ3')
    Shimmer_APQ5 = st.text_input('Shimmer_APQ5')
    MDVP_APQ = st.text_input('MDVP:APQ')
    Shimmer_DDA = st.text_input('Shimmer:DDA')
    NHR = st.text_input('NHR')
    HNR = st.text_input('HNR')
    RPDE = st.text_input('RPDE')
    DFA =  st.text_input('DFA')
    spread1 = st.text_input('spread1')
    spread2 = st.text_input('spread2')
    D2 = st.text_input('D2')
    PPE = st.text_input('PPE')
    
    
    # code for Prediction
    diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Parkinsons Test Result'):
        diagnosis = parkinsons_prediction([MDVP_Fo_Hz, MDVP_Fhi_Hz, MDVP_Flo_Hz, MDVP_Jitter_percentage, MDVP_Jitter_Abs, MDVP_RAP, MDVP_PPQ, Jitter_DDP, MDVP_Shimmer, MDVP_Shimmer_dB, Shimmer_APQ3, Shimmer_APQ5,   MDVP_APQ, Shimmer_DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE])
        
        
    st.success(diagnosis)
    
    
    
    
    
if __name__ == '__main__':
    main()
    
  