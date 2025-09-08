import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load trained model
model = joblib.load("final_model.pkl")


# Title
st.title("Heart Disease Prediction App")
st.write("أدخل بياناتك الصحية علشان نقدر نتنبأ إذا كان في احتمال إصابة بمرض القلب.")


# User Inputs
age = st.number_input("Age", 20, 100, 50)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type (0=typical, 1=atypical, 2=non-anginal, 3=asymptomatic)", [0,1,2,3])
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0,1])
restecg = st.selectbox("Resting ECG Results", [0,1,2])
thalach = st.number_input("Max Heart Rate Achieved", 70, 250, 150)
exang = st.selectbox("Exercise Induced Angina", [0,1])
oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 6.0, 1.0, step=0.1)
slope = st.selectbox("Slope of ST segment", [0,1,2])
ca = st.selectbox("Number of Major Vessels (0-3)", [0,1,2,3])
thal = st.selectbox("Thalassemia (0=normal, 1=fixed defect, 2=reversible defect, 3=other)", [0,1,2,3])

# Convert inputs to numpy array
sex_val = 1 if sex == "Male" else 0
input_data = np.array([[age, sex_val, cp, trestbps, chol, fbs, restecg,
                        thalach, exang, oldpeak, slope, ca, thal]])


# Prediction
if st.button("🔍 Predict"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.error("⚠️ النموذج يتوقع إن في احتمال إصابة بمرض القلب.")
    else:
        st.success("✅ النموذج يتوقع إن مفيش إصابة بمرض القلب.")


# Data Visualization
st.subheader("📊 Explore Heart Disease Dataset")

df = pd.read_csv('processed.cleveland.data', names=[
    "age","sex","cp","trestbps","chol","fbs","restecg",
    "thalach","exang","oldpeak","slope","ca","thal","target"
], na_values="?")
df = df.dropna()
df["target_binary"] = (df["target"] > 0).astype(int)

# Show sample
st.write("Sample of dataset:", df.head())

# Age distribution
fig, ax = plt.subplots()
df["age"].hist(bins=20, ax=ax)
ax.set_title("Age Distribution")
ax.set_xlabel("Age")
ax.set_ylabel("Count")
st.pyplot(fig)

# Cholesterol vs Heart Disease
fig2, ax2 = plt.subplots()
df.groupby("target_binary")["chol"].mean().plot(kind="bar", ax=ax2)
ax2.set_title("Average Cholesterol by Heart Disease")
ax2.set_xlabel("Heart Disease (0=No, 1=Yes)")
ax2.set_ylabel("Average Cholesterol")
st.pyplot(fig2)
