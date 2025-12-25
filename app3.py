import streamlit as st
import pandas as pd
import joblib
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "extra_trees_model.pkl")
scaler_path = os.path.join(current_dir, "scaler.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

st.title("IoT Attack Detection")
st.write("Application for classifying IoT network attack types")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df_new = pd.read_csv(uploaded_file)

    st.subheader("Data Preview")
    st.dataframe(df_new.head())

    if "Unnamed: 0" in df_new.columns:
        df_new = df_new.drop(columns=["Unnamed: 0"])

    try:
        expected_columns = scaler.feature_names_in_
        missing_columns = set(expected_columns) - set(df_new.columns)
        extra_columns = set(df_new.columns) - set(expected_columns)

        if missing_columns:
            st.error(f" Missing columns in the file: {list(missing_columns)}")
            st.stop()

        if extra_columns:
            st.warning(f" Ignored columns (not used by the model): {list(extra_columns)}")
        df_for_prediction = df_new[expected_columns]

        df_scaled = scaler.transform(df_for_prediction)
        predictions = model.predict(df_scaled)
        df_new["Predicted_Attack_Type"] = predictions

        st.subheader("Prediction Results")
        st.dataframe(df_new.head())
        st.subheader("Distribution of Detected Attack Types")
        prediction_counts = pd.Series(predictions).value_counts()
        st.bar_chart(prediction_counts)

        csv = df_new.to_csv(index=False).encode("utf-8")
        st.download_button(
            " Download results",
            csv,
            "predictions.csv",
            "text/csv",
            key="download-csv"
        )

        st.success(f" Prediction successful for {len(df_new)} records")

    except AttributeError:
        st.error(" The scaler does not contain feature names. Please retrain the model using a recent version of scikit-learn.")
        st.info("Attempting prediction using the current column order...")

        df_scaled = scaler.transform(df_new)
        predictions = model.predict(df_scaled)
        df_new["Predicted_Attack_Type"] = predictions

        st.subheader("Prediction Results")
        st.dataframe(df_new.head())

        csv = df_new.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download results",
            csv,
            "predictions.csv",
            "text/csv"
        )

    except Exception as e:
        st.error(f" Prediction error: {str(e)}")
        st.write("Debugging information:")
        st.write(f"- Number of columns in the file: {len(df_new.columns)}")
        st.write(f"- File columns: {df_new.columns.tolist()}")
