import numpy as np
import pandas as pd
import pickle as pkl
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

# Set Streamlit page configuration
st.set_page_config(page_title='Medical Insurance Premium Predictor', layout='wide')

# Load your pre-trained model
@st.cache_resource
def load_model():
    with open('insurancemodel.pkl', 'rb') as file:
        return pkl.load(file)

model = load_model()

# App Header
st.header('Medical Insurance Premium Predictor')
st.subheader("""
This project helps insurance companies in pricing strategies and individuals in understanding their potential insurance costs based on their personal and health characteristics.
""")

# Input Fields
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox('Choose Gender', ['Female', 'Male'])
    smoker = st.selectbox('Are you a smoker?', ['Yes', 'No'])
    age = st.slider('Enter Age', 5, 80, step=1)

with col2:
    bmi = st.slider('Enter BMI', 5.0, 100.0, step=0.1)
    children = st.slider('Choose Number of Children', 0, 5, step=1)

# Prediction button
if st.button('Predict'):
    try:
        # Convert input data into appropriate numerical values
        gender_num = 0 if gender == 'Female' else 1
        smoker_num = 1 if smoker == 'Yes' else 0

        # Prepare input data and make predictions
        input_data = np.array([age, gender_num, bmi, children, smoker_num]).reshape(1, -1)  # 5 features
        predicted_prem = model.predict(input_data)
        display_string = f'**Insurance Premium:** {round(predicted_prem[0], 2)} USD Dollars'
        
        st.success(display_string)
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# Analytics Section
st.header('Analytics on Medical Insurance Dataset')

# Load your dataset for analytics
@st.cache_data
def load_data():
    try:
        return pd.read_csv('insurance.csv')
    except FileNotFoundError:
        st.error("The 'insurance.csv' file was not found. Please ensure it's in the correct directory.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the dataset: {e}")
        return None

data = load_data()

if data is not None:
    # Option to choose the type of analytics
    option = st.selectbox('Choose Analytics Option', ['None', 'Show Data Distribution', 'Show Correlation Heatmap'])

    if option == 'Show Data Distribution':
        # Data distribution for Age, BMI, and Charges
        st.subheader('Distribution of Age, BMI, and Charges')
        
        # Ensure the necessary columns exist
        required_columns = ['age', 'bmi', 'charges']
        if all(col in data.columns for col in required_columns):
            fig, ax = plt.subplots(1, 3, figsize=(18, 5))
            
            sns.histplot(data['age'], ax=ax[0], kde=True, color='blue')
            ax[0].set_title('Age Distribution')
            
            sns.histplot(data['bmi'], ax=ax[1], kde=True, color='green')
            ax[1].set_title('BMI Distribution')
            
            sns.histplot(data['charges'], ax=ax[2], kde=True, color='red')
            ax[2].set_title('Charges Distribution')
            
            st.pyplot(fig)
        else:
            st.error("The dataset does not contain all the required columns for distribution plots.")

    elif option == 'Show Correlation Heatmap':
        # Correlation heatmap
        st.subheader('Correlation Heatmap')
        
        # Select only numeric columns for correlation
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.shape[1] < 2:
            st.error("Not enough numeric columns to compute correlations.")
        else:
            corr_matrix = numeric_data.corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
            ax.set_title('Correlation Heatmap')
            st.pyplot(fig)


