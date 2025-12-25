import streamlit as st
import pandas as pd
import pickle
import numpy as np
from datetime import datetime

st.set_page_config(
    page_title="IoT Attack Detection",
    page_icon="mylogo2.ico",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    try:
        model = pickle.load(open('model.pkl', 'rb'))
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))
        return model, scaler, label_encoder
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

model, scaler, le = load_models()

st.markdown('<h1 class="main-header"> IoT Attack Detection System</h1>', unsafe_allow_html=True)
st.markdown("---")

with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/security-checked.png", width=100)
    st.title("About This App")
    st.info("""
    This application uses Machine Learning to detect and classify cyber attacks in IoT networks.
    
    **Capabilities:**
    - Detect various attack types
    - Process multiple records at once
    - Real-time predictions
    - Export results
    """)
    
    st.markdown("---")
    st.subheader(" Model Information")
    st.write("**Algorithm:** Random Forest")
    st.write("**Accuracy:** 99.5%")
    st.write("**Dataset:** RT-IoT2022")
    
    st.markdown("---")
    st.subheader(" Developer")
    st.write("**Student:** Aya Es ")
    st.write("**Contact:** contact.es.ayah@gmail.com ")
    st.write("**Course:** Data Science , ML ")
    
    st.markdown("---")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d')}")

if model is None:
    st.error("Models could not be loaded. Please ensure all .pkl files are present.")
else:
    tab1, tab2, tab3 = st.tabs([" File Upload", " Manual Input", " Information"])
    
    with tab1:
        st.markdown('<h2 class="sub-header">Upload IoT Network Data</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            <div class="info-box">
            <b>Instructions:</b>
            <ol>
                <li>Prepare your data in CSV or TXT format</li>
                <li>Ensure the data has the same features as the training data</li>
                <li>Upload the file using the uploader below</li>
                <li>Click 'Detect Attacks' to get predictions</li>
            </ol>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.metric("Supported Formats", "CSV, TXT")
            st.metric("Max File Size", "200 MB")
        
        uploaded_file = st.file_uploader("Choose a file", type=['csv', 'txt'], help="Upload your IoT network data file")
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                
                st.success(f"File uploaded  Shape: {df.shape}")
                
                with st.expander(" Data Preview", expanded=True):
                    st.dataframe(df.head(10), use_container_width=True)
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Rows", df.shape[0])
                    col2.metric("Total Columns", df.shape[1])
                    col3.metric("Memory Usage", f"{df.memory_usage().sum() / 1024:.2f} KB")
                
                st.markdown('<h3 class="sub-header">Preprocessing Data...</h3>', unsafe_allow_html=True)
                
                target_columns = ['Attack_type', 'attack_type', 'label', 'target']
                for col in target_columns:
                    if col in df.columns:
                        df = df.drop(col, axis=1)
                        st.info(f"Removed target column: {col}")
                
                if df.isnull().sum().sum() > 0:
                    st.warning("Missing values detected. Filling with mean...")
                    df = df.fillna(df.mean(numeric_only=True))
                
                categorical_cols = df.select_dtypes(include=['object']).columns
                if len(categorical_cols) > 0:
                    st.info(f"Encoding categorical columns: {list(categorical_cols)}")
                    from sklearn.preprocessing import LabelEncoder
                    for col in categorical_cols:
                        le_temp = LabelEncoder()
                        df[col] = le_temp.fit_transform(df[col].astype(str))
                
                try:
                    df_scaled = scaler.transform(df)
                    st.success("Data preprocessed ")
                except Exception as e:
                    st.error(f"Error during scaling: {e}")
                    st.stop()
                
                if st.button('Detect Attacks', type="primary", use_container_width=True):
                    with st.spinner('Analyzing network traffic...'):
                        try:
                            predictions = model.predict(df_scaled)
                            predicted_labels = le.inverse_transform(predictions)
                            
                            df['Predicted_Attack_Type'] = predicted_labels
                            
                            st.markdown('<h3 class="sub-header"> Detection Results</h3>', unsafe_allow_html=True)
                            
                            col1, col2, col3 = st.columns(3)
                            
                            attack_counts = pd.Series(predicted_labels).value_counts()
                            total = len(predicted_labels)
                            normal_count = attack_counts.get('Normal', 0)
                            attack_count = total - normal_count
                            
                            col1.metric("Total Records", total)
                            col2.metric("Normal Traffic", normal_count, delta=f"{(normal_count/total)*100:.1f}%")
                            col3.metric("Attacks Detected", attack_count, delta=f"{(attack_count/total)*100:.1f}%", delta_color="inverse")
                            
                            st.markdown("#### Attack Type Distribution")
                            attack_df = pd.DataFrame({
                                'Attack Type': attack_counts.index,
                                'Count': attack_counts.values,
                                'Percentage': (attack_counts.values / total * 100).round(2)
                            })
                            
                            col1, col2 = st.columns([1, 1])
                            
                            with col1:
                                st.dataframe(attack_df, use_container_width=True, hide_index=True)
                            
                            with col2:
                                st.bar_chart(attack_counts)
                            
                            with st.expander(" View All Predictions", expanded=False):
                                st.dataframe(df, use_container_width=True)
                            
                            csv = df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="Download Results as CSV",
                                data=csv,
                                file_name=f'attack_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                                mime='text/csv',
                                use_container_width=True
                            )
                            
                        except Exception as e:
                            st.error(f"Error during prediction: {e}")
                
            except Exception as e:
                st.error(f"Error reading file: {e}")
    
    with tab2:
        st.markdown('<h2 class="sub-header">Manual Feature Input</h2>', unsafe_allow_html=True)
        
        st.info("Enter network traffic features manually for single instance prediction")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Flow Features")
            flow_duration = st.number_input('Flow Duration', value=0.0, format="%.6f")
            fwd_pkts = st.number_input('Forward Packets Total', value=0, step=1)
            bwd_pkts = st.number_input('Backward Packets Total', value=0, step=1)
        
        with col2:
            st.subheader("Packet Features")
            packet_length_mean = st.number_input('Packet Length Mean', value=0.0, format="%.2f")
            packet_length_std = st.number_input('Packet Length Std', value=0.0, format="%.2f")
            packet_length_max = st.number_input('Packet Length Max', value=0.0, format="%.2f")
        
        with col3:
            st.subheader("Time Features")
            flow_iat_mean = st.number_input('Flow IAT Mean', value=0.0, format="%.6f")
            flow_iat_std = st.number_input('Flow IAT Std', value=0.0, format="%.6f")
            flow_iat_max = st.number_input('Flow IAT Max', value=0.0, format="%.6f")
        
        st.markdown("---")
        st.caption("Note: This is a simplified input. Full model requires 83 features.")
        
        if st.button(' Predict Attack Type', type="primary", use_container_width=True):
            st.warning("Manual prediction requires all 83 features. This is a demonstration only.")
            st.info("Please use the file upload option for accurate predictions.")
    
    with tab3:
        st.markdown('<h2 class="sub-header">About This Project</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ###  Project Overview
            
            This system is designed to detect and classify cyber attacks in IoT networks using machine learning.
            
            **Key Features:**
            - Multi-class attack classification
            - Real-time prediction
            - Batch processing capability
            - High accuracy (>99%)
            
            ### Attack Types Detected
            
            - **DDoS:** Distributed Denial of Service
            - **SSH Brute Force:** Password cracking attempts
            - **Nmap Scan:** Network reconnaissance
            - **Normal:** Legitimate traffic
            - And more...
            """)
        
        with col2:
            st.markdown("""
            ### ⚙️ Technical Details
            
            **Dataset:** RT-IoT2022
            - 123,117 instances
            - 83 features
            - Real IoT infrastructure data
            
            **Model Pipeline:**
            1. Data preprocessing
            2. Feature scaling
            3. Model prediction
            4. Result interpretation
            
            **Performance Metrics:**
            - Accuracy: 99.5%
            - Precision: 99.3%
            - Recall: 99.4%
            - F1-Score: 99.3%
            """)
        
        st.markdown("---")
        
        st.markdown("""
        ### References
        
        - **Dataset Source:** [RT-IoT2022 ]
        - **Institution:** EHTP - École Hassania des Travaux Publics
        - **Program:** Data Science
        - **Course:** Machine Learning Mini-Project 2
        
        ### Acknowledgments
        
        Special thanks to Prof. Abdelhamid FADIL for guidance and support throughout this project.
        """)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem 0;'>
    <p>IoT Attack Detection System | EHTP Data Science , ML  | 2025-2026</p>
    <p>Developed by Aya Es| Mini-projet N°2</p>
</div>
""", unsafe_allow_html=True)
