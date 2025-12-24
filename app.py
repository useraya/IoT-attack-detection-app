import streamlit as st
import pandas as pd
import pickle
import numpy as np
@st.cache_resource
def load_models():
    model = pickle.load(open('model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))
    return model, scaler, label_encoder

model, scaler, le = load_models()

st.title('IoT Attack Detection System')
st.write('Upload IoT network data to detect attack types')

st.sidebar.header('About')
st.sidebar.info('This app detects IoT network attacks using Machine Learning')

uploaded_file = st.file_uploader("Upload CSV/TXT file", type=['csv', 'txt'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.dataframe(df.head())
    
    if 'Attack_type' in df.columns:
        df = df.drop('Attack_type', axis=1)

    df_scaled = scaler.transform(df)
    
    if st.button('Detect Attacks'):
        predictions = model.predict(df_scaled)
        predicted_labels = le.inverse_transform(predictions)
        
        df['Predicted_Attack'] = predicted_labels
        st.success(' Prediction Complete!')
        st.dataframe(df[['Predicted_Attack']])
        
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download Results",
            data=csv,
            file_name='predictions.csv',
            mime='text/csv'
        )

st.subheader('Or Enter Values Manually')
col1, col2 = st.columns(2)
with col1:
    feature1 = st.number_input('Feature 1', value=0.0)
with col2:
    feature2 = st.number_input('Feature 2', value=0.0)


if st.button('Predict Single Instance'):
    input_data = np.array([[feature1, feature2]]) 
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    result = le.inverse_transform(prediction)
    
    st.success(f'Predicted Attack Type: **{result[0]}**')