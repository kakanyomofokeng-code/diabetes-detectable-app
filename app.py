import numpy as np
import pandas as np
import joblib
import streamlit as st

#Load the Model
model = joblib.load('diabetes_model.pkl')

#Set Title of the application

st.title("Diabetes Detection System")
st.write("Enter details of patient ")

#Pregnancies, Glucose, Blood Pressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age

Pregnancies = st.number_input("Pregnancies : ", min_value = 0)

Glucose     = st.number_input("Glucode Level : ")

BloodPressure = st.number_input("Blood Pressure BP : " )

SkinThickness = st.number_input("SkinThickness : ")

Insulin = st.number_input("Insulin : ")

BMI = st.number_input("BMI : ", min_value= 10)

DiabetesPedigreeFunction = st.number_input("DiabetesPedigreeFunction : ")

Age =st.number_input("Age : ", min_value = 18)

 

if st.button("Predict") :

 

    input_data = np.array([Pregnancies,

                           Glucose,

                           BloodPressure,

                           SkinThickness,

                           Insulin,

                           BMI,

                           DiabetesPedigreeFunction,

                           Age ]).reshape(1, -1)

   

    prediction = model.predict(input_data)

 

    if prediction[0] == 1 :

        st.error(" High Risk of Diabetes ")

    else :

        st.success(" Low Risk of Diabetes " )
        

