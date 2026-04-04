import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

# ===================== Page Config =====================
st.set_page_config(page_title="Heart Disease Detector", layout="wide")

# ===================== Styling =====================
st.markdown("""
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1588776814546-ec7e8d7c2b1f");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    .main {
        background-color: rgba(255,255,255,0.85);
        padding: 20px;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 18px;
    }
    .stButton>button:hover {
        background-color: #ff1c1c;
    }
    h1, h2, h3, h4 {
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# ===================== Title =====================
st.title("❤️ Heart Disease Prediction System")
st.write("### Predict risk of heart disease using Machine Learning")

# ===================== Load Model =====================
model = joblib.load("heart_disease_prediction_model.pkl")

# ===================== Sidebar =====================
st.sidebar.header("🧾 Patient Input Features")

age = st.sidebar.slider("Age", 20, 100, 50)
sex = st.sidebar.selectbox("Sex", ["Female", "Male"])
cp = st.sidebar.selectbox("Chest Pain Type", [0,1,2,3])
trestbps = st.sidebar.slider("Resting Blood Pressure", 80, 200, 120)
chol = st.sidebar.slider("Cholesterol", 100, 400, 200)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120", [0,1])
restecg = st.sidebar.selectbox("Rest ECG", [0,1,2])
thalach = st.sidebar.slider("Max Heart Rate", 60, 220, 150)
exang = st.sidebar.selectbox("Exercise Induced Angina", [0,1])
oldpeak = st.sidebar.slider("Oldpeak", 0.0, 6.0, 1.0)
slope = st.sidebar.selectbox("Slope", [0,1,2])
ca = st.sidebar.selectbox("Major Vessels", [0,1,2,3])
thal = st.sidebar.selectbox("Thal", [1,2,3])

# Convert categorical
sex = 1 if sex == "Male" else 0

# ===================== Patient Info + Tips =====================
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📋 Patient Summary")
    st.write(f"**Age:** {age}")
    st.write(f"**Sex:** {'Male' if sex==1 else 'Female'}")
    st.write(f"**Cholesterol:** {chol}")
    st.write(f"**Blood Pressure:** {trestbps}")
    st.write(f"**Max Heart Rate:** {thalach}")

with col2:
    st.markdown("### 💡 General Health Tips")
    st.info("""
    - Maintain healthy diet 🥗  
    - Exercise regularly 🏃  
    - Avoid smoking 🚭  
    - Regular health checkups 🏥  
    """)

# ===================== Predict Button =====================
if st.button("🔍 Predict Heart Disease Risk"):

    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])
    
    prediction = model.predict(input_data)

    # Risk %
    try:
        prob = model.predict_proba(input_data)[0][1]
        risk_percent = round(prob * 100, 2)
    except:
        risk_percent = 50

    st.subheader("📊 Prediction Result")

    # ===================== Risk Meter =====================
    st.markdown("### ❤️ Heart Disease Risk Meter")
    fig, ax = plt.subplots(figsize=(5,0.4))
    gradient = np.linspace(0, 1, 500).reshape(1, -1)
    ax.imshow(gradient, aspect='auto', cmap='RdYlGn_r', extent=[0,100,0,1])
    ax.axvline(risk_percent, color='black', linewidth=2.5)
    ax.set_xlim(0,100)
    ax.set_yticks([])
    ax.set_xlabel("Heart Disease Risk (%)")
    for spine in ax.spines.values():
        spine.set_visible(False)
    st.pyplot(fig)
    st.markdown(f"## ❤️ Risk Score: **{risk_percent}%**")

    # ===================== Doctor Advice =====================
    st.markdown("### 🩺 Doctor’s Advice (AI Assistant)")
    if risk_percent < 30:
        st.info("""
        👨‍⚕️ **Dr. AI:**  
        Good news! Your heart disease risk is low. Keep up your healthy habits — regular exercise, balanced diet, and routine checkups will help maintain your heart health.
        """)
    elif 30 <= risk_percent <= 60:
        st.warning("""
        👨‍⚕️ **Dr. AI:**  
        Your risk is moderate. Focus on improving your lifestyle — reduce cholesterol intake, manage blood pressure, and increase cardiovascular activity. Regular checkups are recommended.
        """)
    else:
        st.error("""
        👨‍⚕️ **Dr. AI:**  
        ⚠️ High risk detected! Consult a cardiologist immediately. Follow medical advice, consider preventive medications as recommended, and strictly manage your diet and exercise.
        """)

    # ===================== Key Metrics Bar Chart =====================
    st.markdown("### 📊 Key Health Metrics")
    fig2, ax2 = plt.subplots(figsize=(5,1.5))
    factors = ["Cholesterol", "Blood Pressure", "Max Heart Rate"]
    values = [chol, trestbps, thalach]
    colors = ['#ff4b4b','#f4a261','#2a9d8f']
    ax2.barh(factors, values, color=colors, height=0.4)
    ax2.set_xlim(0, max(values)+50)
    ax2.set_xlabel("Measurement")
    ax2.set_title("Important Health Metrics")
    for i, v in enumerate(values):
        ax2.text(v + 2, i, str(v), color='black', fontweight='bold')
    st.pyplot(fig2)

    # ===================== Dynamic Recommendations =====================
    st.markdown("### 💡 Personalized Recommendations")

    recommendations = []
    food_suggestions = []
    medicine_suggestions = []

    # Cholesterol
    if chol < 200:
        recommendations.append("✅ Cholesterol is normal.")
        food_suggestions.append("🥗 Continue with fruits, vegetables, and whole grains.")
    elif 200 <= chol <= 239:
        recommendations.append("⚠️ Borderline high cholesterol. Consider diet and exercise.")
        food_suggestions.append("🥦 Increase fiber-rich foods, reduce fried & fatty foods.")
        medicine_suggestions.append("💊 Discuss statins with your doctor if recommended.")
    else:
        recommendations.append("❌ High cholesterol! Follow heart-friendly diet and consult doctor.")
        food_suggestions.append("🥦 Focus on oats, nuts, fruits, vegetables; avoid red meat & processed food.")
        medicine_suggestions.append("💊 Statins may be advised by a doctor.")

    # Blood Pressure
    if trestbps < 120:
        recommendations.append("✅ Blood pressure is normal.")
        food_suggestions.append("💧 Maintain balanced salt intake and hydration.")
    elif 120 <= trestbps <= 129:
        recommendations.append("⚠️ Elevated blood pressure. Reduce salt and exercise.")
        food_suggestions.append("🥗 Limit sodium, increase potassium-rich foods (bananas, spinach).")
        medicine_suggestions.append("💊 If persistent, doctor may prescribe lifestyle-based medications.")
    elif 130 <= trestbps <= 139:
        recommendations.append("❌ Stage 1 Hypertension! Consult doctor and monitor BP.")
        food_suggestions.append("🥗 Low-sodium diet, avoid processed food.")
        medicine_suggestions.append("💊 Antihypertensive medication may be needed under supervision.")
    else:
        recommendations.append("❌ High Blood Pressure! Medical attention recommended.")
        food_suggestions.append("🥗 Strict low-sodium, heart-healthy diet.")
        medicine_suggestions.append("💊 Immediate medical guidance required.")

    # Max Heart Rate
    if thalach >= 100:
        recommendations.append("✅ Max heart rate is healthy.")
        food_suggestions.append("🏃 Maintain regular cardiovascular exercise.")
    else:
        recommendations.append("⚠️ Low max heart rate. Improve cardiovascular fitness with exercise.")
        food_suggestions.append("🏃 Engage in aerobic activities like brisk walking, cycling, swimming.")

    # Prediction-based Recommendations
    if prediction[0] == 1:
        recommendations.append("⚠️ High risk of heart disease. Immediate lifestyle changes and medical consultation advised.")
    else:
        recommendations.append("✅ Low risk of heart disease. Maintain healthy lifestyle.")

    # Display recommendations
    st.markdown("#### 🔹 Health Metrics Recommendations:")
    for rec in recommendations:
        if "❌" in rec:
            st.error(rec)
        elif "⚠️" in rec:
            st.warning(rec)
        else:
            st.success(rec)

    st.markdown("#### 🥗 Food Suggestions:")
    for food in food_suggestions:
        st.info(food)

    st.markdown("#### 💊 Primary Medicine / Preventive Guidance:")
    for med in medicine_suggestions:
        st.warning(med)

# ===================== Footer =====================
st.markdown("---")
st.markdown("👨‍💻 Developed by Chetan | ML Project | Fully Interactive & Interview Ready 🚀")