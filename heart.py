import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open('C:/Users/Lenovo/Desktop/deploy/heart_model.sav', 'rb'))

def heart_prediction(input_data):
    

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return 'The person does not have Heart Disease'
    else:
      return 'The person have Heart Disease'
def main():
    
    
    # giving a title
    st.title('Heart Diseases Prediction using Ml')
    
    
    # getting the input data from the user
    
    
    Age = st.number_input('Enter your Age')
    Sex = st.number_input('SEX')
    CP = st.number_input('Constrictive pericarditis (CP)')
    Trestbps = st.number_input('Trestbps value')
    Chol = st.number_input('Cholesterol Level')
    Fbs = st.number_input('Fasting blood sugar Level')
    Restecg = st.number_input('Resting electrocardiographic result')
    Thalach = st.number_input('Maximum heart rate')
    Exang = st.number_input('Exercise induced angina')
    Oldpeak = st.number_input('ST depression induced by exercise relative to rest.(Oldpeak)')
    Slope = st.number_input('Slope')
    Ca = st.number_input('Coronary Artery')
    Thal = st.number_input('Thalassemia')
    
    # code for Prediction
    diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Test Result'):
        diagnosis = heart_prediction([Age, Sex, CP, Trestbps, Chol, Fbs, Restecg, Thalach, Exang, Oldpeak, Slope, Ca, Thal])        
        
    st.success(diagnosis)
    
    
    
    
    
if __name__ == '__main__':
    main()