# app.py

import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings

# Ignore warnings
warnings.filterwarnings('ignore')

# Set Streamlit page configuration
st.set_page_config(page_title="Diabetes Prediction with Random Forest", page_icon="🩸", layout="centered", initial_sidebar_state="expanded")

# Set up dashboard title and description
st.markdown("<h1 style='text-align: center; color: #3E7A3E;'>🩸 Diabetes Prediction Dashboard (Random Forest)</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>An elegant dashboard powered by a Random Forest Classifier to predict diabetes.</p>", unsafe_allow_html=True)

# Load dataset and preprocess
@st.cache_data
def load_data():
    df = pd.read_csv('diabetes_prediction_dataset.csv')
    
    # Remove duplicates
    df.drop_duplicates(keep='first', inplace=True)
    
    return df

df = load_data()

# Encode categorical features
def preprocess_data(df):
    encode_gender = LabelEncoder()
    encode_smoking = LabelEncoder()
    
    # Applying encodings for gender and smoking history
    df['gender'] = encode_gender.fit_transform(df['gender'])
    df['smoking_history'] = encode_smoking.fit_transform(df['smoking_history'])
    
    return df

df = preprocess_data(df)

# Split data into features and target
x = df.iloc[:, :-1]
y = df['diabetes']

# Load the saved Random Forest model
@st.cache_resource
def load_model():
    return joblib.load('random_forest_model.pkl')

rf = load_model()

# Sidebar - User input for predictions
st.sidebar.header("Enter Patient Data")

def user_input_features():
    # Input for Gender
    gender = st.sidebar.selectbox('Gender', ['Female', 'Male'])
    
    # Input for Age with maximum limit of 80
    age = st.sidebar.slider('Age', 1, 80, 25)
    
    # Input for Hypertension
    hypertension = st.sidebar.selectbox('Hypertension', ['No', 'Yes'])
    
    # Input for Heart Disease
    heart_disease = st.sidebar.selectbox('Heart Disease', ['No', 'Yes'])
    
    # Input for Smoking History
    smoking_history = st.sidebar.selectbox('Smoking History', ['never', 'No_Info', 'current', 'former', 'ever', 'not_current'])
    
    # Input for BMI with maximum limit of 96
    bmi = st.sidebar.slider('BMI', 10, 96, 25)
    
    # Input for HbA1c Level with maximum limit of 9
    HbA1c_level = st.sidebar.slider('HbA1c Level', 3.0, 9.0, 5.0)
    
    # Input for Blood Glucose Level with maximum limit of 300
    blood_glucose_level = st.sidebar.slider('Blood Glucose Level', 50, 300, 100)
    
    # Encode inputs for model
    gender = 1 if gender == 'Male' else 0
    smoking_history_encoded = {'never': 0, 'No_Info': 1, 'current': 2, 'former': 3, 'ever': 4, 'not_current': 5}[smoking_history]
    hypertension_encoded = 1 if hypertension == 'Yes' else 0
    heart_disease_encoded = 1 if heart_disease == 'Yes' else 0
    
    # Create dataframe of inputs
    data = {
        'gender': gender,
        'age': age,
        'hypertension': hypertension_encoded,
        'heart_disease': heart_disease_encoded,
        'smoking_history': smoking_history_encoded,
        'bmi': bmi,
        'HbA1c_level': HbA1c_level,
        'blood_glucose_level': blood_glucose_level
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

# Store user input
input_df = user_input_features()

# Display the input table in the main section
st.markdown("<h2 style='text-align: center;'>Patient Data Input</h2>", unsafe_allow_html=True)
st.table(input_df)

# Button for prediction in the main section
predict_btn = st.button('Predict')

# Predict and display results on button click
if predict_btn:
    
    # Make predictions
    prediction = rf.predict(input_df)
    prediction_proba = rf.predict_proba(input_df)
    
    # Display the prediction result in green or red
    st.markdown("<h2 style='text-align: center;'>Prediction Result</h2>", unsafe_allow_html=True)
    
    if prediction[0] == 1:
        st.markdown(f"<h3 style='color: red; text-align: center;'>Patient is likely to have diabetes.</h3>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h3 style='color: green; text-align: center;'>Patient is unlikely to have diabetes.</h3>", unsafe_allow_html=True)
    
    # Display prediction probabilities for diabetes and non-diabetes
    st.markdown("<h2 style='text-align: center;'>Prediction Probability</h2>", unsafe_allow_html=True)
    st.write(f"<p style='text-align: center; color: red'>Probability of having diabetes: <strong>{round(prediction_proba[0][1] * 100, 2)}%</strong></p>", unsafe_allow_html=True)
    st.write(f"<p style='text-align: center; color: green'>Probability of not having diabetes: <strong>{round(prediction_proba[0][0] * 100, 2)}%</strong></p>", unsafe_allow_html=True)

# Footer
st.markdown("""
    <hr>
    <p style='text-align: center;'>Developed with ❤️</p>
""", unsafe_allow_html=True)
