# IoT Attack Detection System

A machine learning-based web application for detecting and classifying cyber attacks in IoT networks using the RT-IoT2022 dataset.

## Project Context

This project is part of the **Machine Learning Mini-Project 2** for the Data Science program at EHTP (Hassania School of Public Works).

- **Student**: Aya Es
- **Course**: Machine Learning , DS
- **Academic Year**: 2025-2026

## Overview

This system uses a Random Forest classifier trained on the RT-IoT2022 dataset to identify and classify various types of network attacks in IoT infrastructure. The application provides an intuitive web interface for real-time attack detection and classification.

## Dataset Information

**RT-IoT2022 Dataset**
- **Source**: Real IoT infrastructure combining normal and malicious traffic
- **Size**: 123,117 instances
- **Features**: 83 network traffic characteristics
- **Data Types**: Mixed (numerical and categorical)
- **Attack Scenarios**: SSH Brute-force, DDoS, Nmap scans, and others
- **Purpose**: Development and evaluation of Intrusion Detection Systems (IDS) in IoT environments
- **Reference**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/942/rt-iot2022)

## Project Components

### 1. Exploratory Data Analysis (Jupyter Notebook)
The `ML_IoT_Attack.ipynb` file contains:
- Problem understanding and ML role in attack detection
- Comprehensive data exploration and visualization
- Variable analysis, correlations, and statistical distributions
- Target variable analysis and class distribution
- Insights and conclusions from EDA

### 2. Data Preprocessing & Model Development
- Missing data handling
- Categorical variable encoding
- Feature scaling and normalization
- Outlier detection and treatment
- Feature selection and engineering
- Training and evaluation of 10 ML algorithms
- Model comparison using various performance metrics

### 3. Model Selection & Tuning
- Selection of top 2 performing models
- Hyperparameter tuning and optimization
- Final model selection with test results
- Performance evaluation (accuracy, precision, recall, F1-score)

### 4. Web Application (Streamlit)
The `finalapp.py` file provides:
- User-friendly interface for attack detection
- Batch processing of network traffic data
- Real-time prediction and classification
- Results visualization and export functionality

## Features

- **File Upload**: Process CSV/TXT files with multiple network traffic records
- **Real-time Detection**: Instant classification of traffic patterns
- **Multi-class Classification**: Detects various attack types
- **Results Visualization**: Interactive charts and statistics
- **Export Functionality**: Download predictions as CSV
- **Model Information**: View algorithm details and performance metrics

## Attack Types Detected

- **Normal**: Legitimate IoT device usage
- **DDoS**: Distributed Denial of Service attacks
- **SSH Brute Force**: Password cracking attempts
- **Nmap Scan**: Network reconnaissance activities
- **Other attack types** based on the RT-IoT2022 dataset

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager


### Using the Web Interface

1. **File Upload Tab**: 
   - Upload CSV/TXT file with network traffic features
   - Click "Run Detection" to analyze
   - View results and download predictions

2. **Manual Input Tab**: 
   - Enter individual feature values for testing
   - Note: Full prediction requires all 83 features

3. **Info Tab**: 
   - Project overview and technical details
   - Model performance metrics

## Model Performance

- **Algorithm**: Random Forest Classifier
- **Accuracy**: ~99.5%
- **Precision**: ~99.3%
- **Recall**: ~99.4%
- **F1-Score**: ~99.3%

## Project Structure

```
iot-attack-detection-app/
├── ML_IoT_Attack.ipynb      
├── finalapp.py              
├── model.pkl                
├── scaler.pkl               
├── label_encoder.pkl        
├── RT-IoT2022.txt           
├── requirements.txt        
├── mylogo.png               
└── README.md              
```

## Dependencies

```
streamlit
pandas
numpy
scikit-learn
matplotlib
seaborn
jupyter
```

Install all dependencies with:
```bash
pip install -r requirements.txt
```

## Deployment

The application is deployed on **Streamlit Cloud Community**.

**Live Application**: https://iot-attack-detection-app-app.streamlit.app/

## Project Methodology

This project follows a complete ML pipeline:

1. **Problem Understanding**: Analysis of IoT attack detection requirements
2. **Exploratory Data Analysis**: Comprehensive data exploration and insights
3. **Data Preprocessing**: Cleaning, encoding, scaling, and feature selection
4. **Model Development**: Training and evaluation of 10+ algorithms
5. **Model Selection**: Comparison and selection of best performers
6. **Hyperparameter Tuning**: Optimization of selected models
7. **Deployment**: Web application development and cloud deployment
8. **Testing**: Validation with new data



## Acknowledgments

Special thanks to **Prof. Abdelhamid FADIL** for guidance and supervision throughout this project.

Dataset acknowledgment: RT-IoT2022 dataset from the UCI Machine Learning Repository.

## License

This project is for educational purposes as part of the EHTP Data Science program.

## Contact

For questions or feedback regarding this project:
- **Email**: [contact.es.ayah@gmail.com]
