import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="Student Dropout Early Warning System",
    page_icon="🎓",
    layout="wide"
)

# =============================
# CUSTOM LIGHT-BLUE THEME
# =============================
st.markdown("""
<style>
body {background-color: #f0f9ff;}
h1,h2,h3,h4 {color:#0284c7;}
.metric-box {
    background-color:#e0f2fe;
    padding:20px;
    border-radius:15px;
    text-align:center;
    border:2px solid #38bdf8;
}
.metric-title {color:#075985;}
.metric-value {
    font-size:30px;
    font-weight:bold;
    color:#0369a1;
}
</style>
""", unsafe_allow_html=True)

# =============================
# HEADER
# =============================
st.title("🎓 Student Dropout Early Warning System")
st.caption("Real-World Risk Insights • Probability-Based Predictions")

# =============================
# FILE UPLOAD
# =============================
uploaded_file = st.file_uploader(
    "📂 Upload Student Dataset (CSV)",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Please upload a student CSV file to continue.")
    st.stop()

# =============================
# LOAD DATA
# =============================
df = pd.read_csv(uploaded_file)

# =============================
# SIDEBAR – DATA OVERVIEW
# =============================
st.sidebar.header("📊 Data Overview")

st.sidebar.write("**Total Rows:**", df.shape[0])
st.sidebar.write("**Total Columns:**", df.shape[1])
st.sidebar.write("**Missing Values:**", df.isnull().sum().sum())
st.sidebar.write("**Duplicate Rows:**", df.duplicated().sum())

st.sidebar.divider()
st.sidebar.subheader("ℹ️ Dataset Information")

st.sidebar.markdown("""
This dataset contains **academic, behavioral, and demographic**
information of students such as:

• Classroom participation  
• LMS resource usage  
• Attendance behavior  
• Semester & subject details  
• Parental involvement  

🎯 **Goal:** Identify students at risk of dropping out **early**  
so institutions can take timely action.
""")

# =============================
# BASIC CLEANING
# =============================
df = df.drop_duplicates()

# =============================
# TARGET CREATION
# =============================
df['dropout'] = df['Class'].map({'L':1, 'M':0, 'H':0})
df.drop(columns=['Class'], inplace=True)

# =============================
# ENCODING
# =============================
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# =============================
# FEATURES & TARGET
# =============================
X = df.drop(columns=['dropout'])
y = df['dropout']

# =============================
# TRAIN TEST SPLIT
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =============================
# MODELS
# =============================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        class_weight="balanced",
        random_state=42
    )
}

scores = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    scores[name] = accuracy_score(y_test, preds)

# =============================
# MODEL PERFORMANCE
# =============================
st.subheader("📈 Model Performance")

perf_df = pd.DataFrame({
    "Model": scores.keys(),
    "Accuracy": scores.values()
})

fig_perf = px.bar(
    perf_df,
    x="Model",
    y="Accuracy",
    color="Model",
    title="Model Accuracy Comparison"
)
st.plotly_chart(fig_perf, use_container_width=True)

# =============================
# FINAL MODEL (Random Forest)
# =============================
final_model = models["Random Forest"]

proba = final_model.predict_proba(X)[:,1]

df_results = df.copy()
df_results["student_id"] = df_results.index
df_results["risk_score"] = proba

def risk_label(p):
    if p >= 0.60:
        return "High"
    elif p >= 0.30:
        return "Medium"
    else:
        return "Low"

df_results["risk_label"] = df_results["risk_score"].apply(risk_label)
df_results["predicted_dropout"] = (df_results["risk_score"] >= 0.5).astype(int)

# =============================
# KPI CARDS
# =============================
st.subheader("📌 Student Risk Overview")

col1,col2,col3 = st.columns(3)

col1.markdown(f"""
<div class="metric-box">
<div class="metric-title">High Risk</div>
<div class="metric-value">{(df_results['risk_label']=="High").sum()}</div>
</div>
""", unsafe_allow_html=True)

col2.markdown(f"""
<div class="metric-box">
<div class="metric-title">Medium Risk</div>
<div class="metric-value">{(df_results['risk_label']=="Medium").sum()}</div>
</div>
""", unsafe_allow_html=True)

col3.markdown(f"""
<div class="metric-box">
<div class="metric-title">Low Risk</div>
<div class="metric-value">{(df_results['risk_label']=="Low").sum()}</div>
</div>
""", unsafe_allow_html=True)

# =============================
# TOP HIGH-RISK STUDENTS
# =============================
st.subheader("🚨 Top 20 High-Risk Students")

top_high = df_results[df_results["risk_label"]=="High"] \
    .sort_values(by="risk_score", ascending=False) \
    .head(20)

st.dataframe(
    top_high[["student_id","risk_score","risk_label","predicted_dropout"]],
    use_container_width=True
)

# =============================
# SINGLE STUDENT VIEW
# =============================
st.subheader("🎯 Individual Student Risk Analysis")

sid = st.selectbox("Select Student ID", df_results["student_id"])
student = df_results[df_results["student_id"]==sid].iloc[0]

c1,c2,c3 = st.columns(3)
c1.metric("Risk Score", f"{student['risk_score']:.2f}")
c2.metric("Risk Level", student["risk_label"])
c3.metric("Predicted Dropout", student["predicted_dropout"])

# =============================
# FEATURE IMPORTANCE
# =============================
st.subheader("🧠 Top Reasons for Dropout")

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": final_model.feature_importances_
}).sort_values(by="Importance", ascending=False).head(10)

fig_imp = px.bar(
    importance_df,
    x="Importance",
    y="Feature",
    orientation="h",
    title="Top Influencing Factors"
)
st.plotly_chart(fig_imp, use_container_width=True)

# =============================
# DOWNLOAD CSV
# =============================
st.subheader("⬇️ Download Prediction Results")

final_csv = df_results[
    ["student_id","risk_score","risk_label","predicted_dropout"]
]

st.download_button(
    "Download CSV",
    final_csv.to_csv(index=False),
    file_name="student_dropout_predictions.csv",
    mime="text/csv"
)
