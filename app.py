import joblib
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Memuat model dan preprocessor dari file
model = joblib.load('model.pkl')

# Definisikan fitur numerik dan kategorikal
numerical_features = ['Age', 'Family size', 'latitude', 'longitude']
categorical_features = ['Gender', 'Marital Status', 'Occupation', 'Monthly Income', 'Educational Qualifications', 'Feedback']

# Preprocessor harus dibuat dan disesuaikan
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Judul Aplikasi
st.title('Aplikasi Prediksi Data Pelanggan')

# Input data
st.subheader('Masukkan Data Pelanggan')
age = st.number_input('Age', min_value=0, max_value=100, value=18)
gender = st.selectbox('Gender', ['Male', 'Female'])
marital_status = st.selectbox('Marital Status', ['Single', 'Married', 'Divorced', 'Widowed'])
occupation = st.selectbox('Occupation', ['Student', 'Employed', 'Unemployed', 'Retired'])
monthly_income = st.selectbox('Monthly Income', ['No Income', 'Low Income', 'Medium Income', 'High Income'])
education = st.selectbox('Educational Qualifications', ['Under Graduate', 'Graduate', 'Post Graduate', 'PhD'])
feedback = st.selectbox('Feedback', ['Positive', 'Negative', 'Neutral'])
family_size = st.number_input('Family size', min_value=1, max_value=20, value=1)
latitude = st.number_input('Latitude', min_value=-90.0, max_value=90.0, value=0.0)
longitude = st.number_input('Longitude', min_value=-180.0, max_value=180.0, value=0.0)
pin_code = st.number_input('Pin code', min_value=0, max_value=999999, value=100000)

# Mapping input ke format yang diharapkan oleh model
input_data = pd.DataFrame({
    'Age': [age],
    'Gender': [gender],
    'Marital Status': [marital_status],
    'Occupation': [occupation],
    'Monthly Income': [monthly_income],
    'Educational Qualifications': [education],
    'Feedback': [feedback],
    'Family size': [family_size],
    'latitude': [latitude],
    'longitude': [longitude],
    'Pin code': [pin_code]
})

# Pra-pemrosesan input menggunakan preprocessor yang sudah ditentukan
input_data_processed = preprocessor.fit_transform(input_data)

# Tombol Prediksi
if st.button('Predict'):
    prediction = model.predict(input_data_processed)
    st.subheader('Output Prediksi')
    st.write(f'Hasil prediksi: {prediction[0]}')
