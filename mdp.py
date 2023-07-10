import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
from PIL import Image


image = Image.open('C:/Users/Lenovo/Desktop/minor/deploy/a1.jpg')
# loading the saved models
diabetes_model = pickle.load(open('C:/Users/Lenovo/Desktop/minor/deploy/model/diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open('C:/Users/Lenovo/Desktop/minor/deploy/model/heart_model.sav', 'rb'))
parkinsons_model = pickle.load(open('C:/Users/Lenovo/Desktop/minor/deploy/model/parkinsons_model.sav', 'rb'))
medical_insurance_cost = pickle.load(open('C:/Users/Lenovo/Desktop/minor/deploy/model/medical_insurance_cost.sav', 'rb'))
bcc_model = pickle.load(open('C:/Users/Lenovo/Desktop/minor/deploy/model/breast_cancer.sav', 'rb'))

# Sidebar for navigation
with st.sidebar:
    
    
    selected = option_menu('Multiple Disease Prediction System',
                           ['Breast Cancer Classification','Heart Disease Prediction','Diabetes Prediction', 'Parkinsons Prediction', ''],
                           icons=['person-bounding-box','heart','activity','person',''],
                           default_index=4)
with st.sidebar:   
    s1 = option_menu('Medical Insurance Prediction System',
                         ['Medical Insurance Cost Prediction',""],
                         icons=['person'], default_index=1)

if (selected == '' and s1 == ""):
    st.image(image, width=350)
    st.title('Multiple Health Care Predictor Using Machine Learning')
    
                         
# Diabetes Prediction Page
if (selected == 'Breast Cancer Classification' and s1 not in ['Medical Insurance Cost Prediction']):
    
    # page title
    st.title("Breast Cancer Classification Using Machine Learning")
    
    
    # getting the input data from the user
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        Radius = st.number_input('Radius Mean')
    with col2:
        Texture = st.number_input('SD of grey scale value')
    with col3:    
        Perimeter = st.number_input('Perimeter')
    with col4:
        Area = st.number_input('Area Mean')
    with col5:
        Smoothness = st.number_input('Smoothness Mean')
    with col1:
        Compactness = st.number_input('Compactness Mean')
    with col2:
        Concavity = st.number_input('Concavity Mean')
    with col3:    
        Concave = st.number_input('Concave Points Mean')
    with col4:
        Symmetry = st.number_input('Symmetry Mean')
    with col5:
        Fractal = st.number_input('Fractal Dim Mean')
    with col1:
        Radius1= st.number_input('Radius se')
    with col2:
        Texture1 = st.number_input('Texture se')
    with col3:    
        Perimeter1 = st.number_input('Perimeter se')
    with col4:
         Area1 = st.number_input('Area se')
    with col5:
        Smoothness1 = st.number_input('Smoothness se')
    with col1:
        Compactness1 = st.number_input('Compactness se')
    with col2:
        Concavity1= st.number_input('Concavity se')
    with col3:
        Concave1 = st.number_input('Concave points se')
    with col4:    
        Symmetry1 = st.number_input('Symmetry se')
    with col5:
        Fractal1 = st.number_input('Fractal dim se')
    with col1:
        Radius2 = st.number_input('Radius worst')
    with col2:
        Texture2 = st.number_input('Texture worst')
    with col3:
        Perimeter2 = st.number_input('Perimeter worst')
    with col4:    
        Area2 = st.number_input('Area worst')
    with col5:
        Smoothness2 = st.number_input('Smoothness worst')
    with col1:
        Compactness2 = st.number_input('Compactness worst')
    with col2:
        Concavity2 = st.number_input('Concavity worst')
    with col3:
        Concave2 = st.number_input('Concave points worst')
    with col4:    
        Symmetry2 = st.number_input('Symmetry worst')
    with col5:
        Fractal2 = st.number_input('Fractal dim worst')
    
    # code for Prediction
    bcc = ''
    
    # creating a button for Prediction
    
    if st.button('Breast Cancer Classification Test Result'):
        bcc1 = bcc_model.predict([[Radius, Texture, Perimeter, Area, Smoothness, Compactness, Concavity, Concave, Symmetry, Fractal, Radius1, Texture1, Perimeter1, Area1, Smoothness1, Compactness1, Concavity1, Concave1, Symmetry1, Fractal1, Radius2, Texture2, Perimeter2, Area2, Smoothness2, Compactness2, Concavity2, Concave2, Symmetry2, Fractal2]])
        
        if (bcc1[0] == 1):
          bcc = 'The Breast Cancer is Benign'
        else:
          bcc = 'The Breast cancer is Malignant'
        
    st.success(bcc)
    
    
# Diabetes Prediction Page
if (selected == 'Diabetes Prediction' and s1 not in ['Medical Insurance Cost Prediction']):
    
    # page title
    st.title('Diabetes Prediction using ML')
    
    
    # getting the input data from the user
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.number_input('Number of Pregnancies')
        
    with col2:
        Glucose = st.number_input('Glucose Level')
    
    with col3:
        BloodPressure = st.number_input('Blood Pressure value')
    
    with col1:
        SkinThickness = st.number_input('Skin Thickness value')
    
    with col2:
        Insulin = st.number_input('Insulin Level')
    
    with col3:
        BMI = st.number_input('BMI value')
    
    with col1:
        DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function value')
    
    with col2:
        Age = st.number_input('Age of the Person')
    
    
    # code for Prediction
    diab_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
        diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        
        if (diab_prediction[0] == 1):
          diab_diagnosis = 'The person is diabetic'
        else:
          diab_diagnosis = 'The person is not diabetic'
        
    st.success(diab_diagnosis)




# Heart Disease Prediction Page
if (selected == 'Heart Disease Prediction' and s1 not in ['Medical Insurance Cost Prediction']):
    
    # page title
    st.title('Heart Disease Prediction using ML')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input('Age')
        
    with col2:
        sex = st.number_input('Sex (1-male/0-female)')
        
    with col3:
        cp = st.number_input('Chest Pain types')
        
    with col1:
        trestbps = st.number_input('Resting Blood Pressure in mm Hg')
        
    with col2:
        chol = st.number_input('Serum Cholestoral in mg/dl')
        
    with col3:
        fbs = st.number_input('Fasting Blood Sugar > 120 mg/dl')
        
    with col1:
        restecg = st.number_input('Resting Electrocardiographic results')
        
    with col2:
        thalach = st.number_input('Maximum Heart Rate achieved')
        
    with col3:
        exang = st.number_input('Exercise Induced Angina')
        
    with col1:
        oldpeak = st.number_input('ST depression induced by exercise')
        
    with col2:
        slope = st.number_input('Slope of the peak exercise ST segment')
        
    with col3:
        ca = st.number_input('Major vessels colored by flourosopy')
        
    with col1:
        thal = st.number_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')
        
        
     
     
    # code for Prediction
    heart_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Heart Disease Test Result'):
        heart_prediction = heart_disease_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg,thalach,exang,oldpeak,slope,ca,thal]])                          
        
        if (heart_prediction[0] == 1):
          heart_diagnosis = 'The person may have  heart disease'
        else:
          heart_diagnosis = 'The person does not have heart disease'
        
    st.success(heart_diagnosis)
        
    
    

# Parkinson's Prediction Page
if (selected == "Parkinsons Prediction" and s1 not in ['Medical Insurance Cost Prediction']):
    
    # page title
    st.title("Parkinson's Disease Prediction using ML")
    
    col1, col2, col3, col4, col5 = st.columns(5)  
    
    with col1:
        fo = st.number_input('MDVP:Fo(Hz)')
        
    with col2:
        fhi = st.number_input('MDVP:Fhi(Hz)')
        
    with col3:
        flo = st.number_input('MDVP:Flo(Hz)')
        
    with col4:
        Jitter_percent = st.number_input('MDVP:Jitter(%)')
        
    with col5:
        Jitter_Abs = st.number_input('MDVP:Jitter(Abs)')
        
    with col1:
        RAP = st.number_input('MDVP:RAP')
        
    with col2:
        PPQ = st.number_input('MDVP:PPQ')
        
    with col3:
        DDP = st.number_input('Jitter:DDP')
        
    with col4:
        Shimmer = st.number_input('MDVP:Shimmer')
        
    with col5:
        Shimmer_dB = st.number_input('MDVP:Shimmer(dB)')
        
    with col1:
        APQ3 = st.number_input('Shimmer:APQ3')
        
    with col2:
        APQ5 = st.number_input('Shimmer:APQ5')
        
    with col3:
        APQ = st.number_input('MDVP:APQ')
        
    with col4:
        DDA = st.number_input('Shimmer:DDA')
        
    with col5:
        NHR = st.number_input('NHR')
        
    with col1:
        HNR = st.number_input('HNR')
        
    with col2:
        RPDE = st.number_input('RPDE')
        
    with col3:
        DFA = st.number_input('DFA')
        
    with col4:
        spread1 = st.number_input('spread1')
        
    with col5:
        spread2 = st.number_input('spread2')
        
    with col1:
        D2 = st.number_input('D2')
        
    with col2:
        PPE = st.number_input('PPE')
        
    
    
    # code for Prediction
    parkinsons_diagnosis = ''
    
    # creating a button for Prediction    
    if st.button("Parkinson's Test Result"):
        parkinsons_prediction = parkinsons_model.predict([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ,DDP,Shimmer,Shimmer_dB,APQ3,APQ5,APQ,DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE]])                          
        
        if (parkinsons_prediction[0] == 1):
          parkinsons_diagnosis = "The person has Parkinson's disease"
        else:
          parkinsons_diagnosis = "The person does not have Parkinson's disease"
        
    st.success(parkinsons_diagnosis)

if (s1 == 'Medical Insurance Cost Prediction' and selected not in ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction']):
    
    st.title('Medical Insurance Cost Prediction using ML')

    col1, col2, col3 = st.columns(3)
    
    with col1:
        Age = st.number_input('Enter your AGE')
        
    with col2:
        Sex = st.number_input('SEX (0-MALE/1-FEMALE)')
    
    with col3:
        BMI = st.number_input('Body mass index value')
    
    with col1:
        Children = st.number_input('Number of Children')
    
    with col2:
        Smoker = st.number_input('Do you smoke? (0-Yes/1-No)')
    
    with col3:
        Region = st.number_input('Enter Your Region (0-southeast/1-southwest/2-northeast/3-northwest)')
        
    
    MICT = ''
    
    # creating a button for Prediction    
    if st.button("Medical Insurance Cost"):
        MICT = np.asarray([[Age, Sex, BMI, Children, Smoker, Region]])
        
        input_data_reshaped = MICT.reshape(1,-1)

        prediction = medical_insurance_cost.predict(input_data_reshaped)
        
        MICT ='The insurance cost is USD: ', prediction[0]
        
  
        
    st.success(MICT)
