import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import warnings

# Ignore warnings
warnings.filterwarnings('ignore')

# Set page title and configuration
st.set_page_config(page_title="Loan Approval Prediction System", page_icon="📊")
st.title("📊 Loan Approval Prediction System")
st.markdown("This app predicts whether your loan application would be approved based on your information.")

# Load train dataset
@st.cache_data
def load_train_data():
    return pd.read_csv("train_u6lujuX_CVtuZ9i.csv")

@st.cache_data
def load_test_data():
    return pd.read_csv("test_Y3wMUE5_7gLdaTN.csv")

try:
    loan_df = load_train_data()
    st.success(f"✅ Training dataset loaded successfully! {loan_df.shape[0]} records.")
    
    with st.expander("View training data sample"):
        st.dataframe(loan_df.head())
except Exception as e:
    st.error(f"Error loading training dataset: {str(e)}")
    st.stop()

# Preprocessing function
def preprocess_data(df):
    df_processed = df.copy()

    for col in ['LoanAmount', 'Loan_Amount_Term', 'Credit_History']:
        if col in df_processed.columns:
            df_processed[col].fillna(df_processed[col].mean(), inplace=True)

    for col in ['Gender', 'Married', 'Dependents', 'Self_Employed']:
        if col in df_processed.columns:
            df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)

    loan_ids = None
    if 'Loan_ID' in df_processed.columns:
        loan_ids = df_processed['Loan_ID']
        df_processed.drop('Loan_ID', axis=1, inplace=True)

    categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 
                        'Self_Employed', 'Property_Area']
    
    encoders = {}
    for col in categorical_cols:
        if col in df_processed.columns:
            encoder = LabelEncoder()
            df_processed[col] = encoder.fit_transform(df_processed[col])
            encoders[col] = encoder

    if 'Loan_Status' in df_processed.columns:
        df_processed['Loan_Status'] = df_processed['Loan_Status'].map({'Y': 1, 'N': 0})

    return df_processed, encoders, loan_ids

# Preprocess training data
loan_df_processed, encoders, loan_ids = preprocess_data(loan_df)
X = loan_df_processed.drop('Loan_Status', axis=1)
y = loan_df_processed['Loan_Status']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
st.success(f"✅ Model trained with accuracy: {accuracy:.2f}")

with st.expander("View validation predictions"):
    val_results = pd.DataFrame({
        'Actual': y_val,
        'Predicted': y_pred,
        'Correct': y_val == y_pred
    })
    st.dataframe(val_results)

# --- TEST SET PREDICTION ---
st.header("📁 Test Dataset Prediction")

try:
    test_data = load_test_data()
    st.success(f"✅ Test dataset loaded successfully! {test_data.shape[0]} records.")
    
    with st.expander("View test data sample"):
        st.dataframe(test_data.head())
    
    test_data_processed = test_data.copy()
    
    for col in ['LoanAmount', 'Loan_Amount_Term', 'Credit_History']:
        test_data_processed[col].fillna(test_data_processed[col].mean(), inplace=True)

    for col in ['Gender', 'Married', 'Dependents', 'Self_Employed']:
        test_data_processed[col].fillna(test_data_processed[col].mode()[0], inplace=True)

    loan_ids_test = test_data_processed['Loan_ID']
    test_data_processed.drop('Loan_ID', axis=1, inplace=True)

    for col in encoders:
        test_data_processed[col] = encoders[col].transform(test_data_processed[col])

    test_predictions = model.predict(test_data_processed)

    test_result_df = pd.DataFrame({
        'Loan_ID': loan_ids_test,
        'Loan_Status_Predicted': ['Y' if pred == 1 else 'N' for pred in test_predictions]
    })

    st.subheader("🔍 Predicted Loan Status for Test Dataset")
    st.dataframe(test_result_df)

    csv = test_result_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Predictions CSV", csv, "loan_predictions.csv", "text/csv")

except Exception as e:
    st.error(f"Error processing test dataset: {str(e)}")

# --- SIDEBAR USER INPUT ---
st.sidebar.header("Enter Your Information")
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
married = st.sidebar.selectbox("Married", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])

# ✅ CHANGED THIS PART: Replaced sliders with number_input
applicant_income = st.sidebar.number_input("Applicant Income", min_value=0, value=int(loan_df['ApplicantIncome'].mean()))
coapplicant_income = st.sidebar.number_input("Coapplicant Income", min_value=0, value=int(loan_df['CoapplicantIncome'].mean()))
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0, value=int(loan_df['LoanAmount'].mean()))

# Everything below remains the same
loan_term = st.sidebar.selectbox("Loan Term (in months)", 
                                sorted(loan_df['Loan_Amount_Term'].dropna().unique().astype(int).tolist()))
credit_history = st.sidebar.selectbox("Credit History (1: has credit history, 0: no credit history)", [1, 0])
property_area = st.sidebar.selectbox("Property Area", loan_df['Property_Area'].unique())

if st.sidebar.button("Predict Loan Approval"):
    user_data = pd.DataFrame({
        'Gender': [gender],
        'Married': [married],
        'Dependents': [dependents],
        'Education': [education],
        'Self_Employed': [self_employed],
        'ApplicantIncome': [applicant_income],
        'CoapplicantIncome': [coapplicant_income],
        'LoanAmount': [loan_amount],
        'Loan_Amount_Term': [loan_term],
        'Credit_History': [credit_history],
        'Property_Area': [property_area]
    })

    user_data_processed = user_data.copy()
    for col in encoders:
        try:
            user_data_processed[col] = encoders[col].transform(user_data_processed[col])
        except ValueError:
            fallback = loan_df[col].mode()[0]
            user_data_processed[col] = fallback
            user_data_processed[col] = encoders[col].transform(user_data_processed[col])
            st.warning(f"Unseen value in {col}. Replaced with: {fallback}")

    prediction = model.predict(user_data_processed)[0]
    probability = model.predict_proba(user_data_processed)[0]

    st.header("Loan Prediction Result")
    if prediction == 1:
        st.success("✅ Congratulations! Your loan is likely to be APPROVED.")
        st.balloons()
    else:
        st.error("❌ Sorry, your loan is likely to be REJECTED.")

    st.write(f"Probability of approval: {probability[1] * 100:.2f}%")

    st.subheader("Key factors in this decision:")
    feature_names = X.columns
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Value': [user_data[col].values[0] if col in user_data.columns else "N/A" for col in feature_names],
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

    st.write(importance_df)
    st.bar_chart(importance_df.set_index('Feature')['Importance'])

# About section
st.markdown("---")
st.markdown("""
## About this app
This loan prediction system uses a Decision Tree model to determine if a loan application is likely to be approved.

### Tips for approval:
- A strong credit history helps significantly
- A higher income-to-loan ratio improves your chances
- Properties in semi-urban or urban areas may improve approval odds
""")
