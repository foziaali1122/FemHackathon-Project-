# 🎓 Student Dropout Early Warning System  
### Real-World Risk Insights • Early Intervention • Data-Driven Decisions  

![Hackathon](https://img.shields.io/badge/Hackathon-Project-blueviolet)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![Plotly](https://img.shields.io/badge/Visualization-Plotly-blue)
![Status](https://img.shields.io/badge/Status-Completed-success)

---

## 🚀 Project Overview  

This project is **not just a model prediction system**.  
It is a **Real-World Student Dropout Early Warning System** designed to help institutions **identify at-risk students early** and take **preventive actions**.

Instead of predicting only *Yes / No*, this system calculates **Dropout Risk Probability**, classifies students into **High / Medium / Low Risk**, and explains **why students are likely to drop out**.

🎯 **Goal:**  
> *Early detection → Timely intervention → Reduced dropout rates*

---

## 🧠 Why This Project is Different  

✔ Real-world **risk probability**, not just accuracy  
✔ **Multiple ML models** compared for stability  
✔ **Interactive Plotly dashboards**  
✔ Individual student-level risk analysis  
✔ Actionable insights for educators  

---

## 🖥️ Dashboard Preview  

> 📸 *Live Streamlit Dashboard*

![Dashboard Screenshot](screenshots/dashboard.png)

---

## 📂 Dataset  

- **xAPI-Edu-Data.csv**
- Educational & behavioral student data  
- Used in real academic research  

---

## 📊 Dashboard Sections  

### 🔹 1. Data Overview  
- Total Students  
- Total Features  
- Missing Values  
- Duplicate Records  
- Data Quality Summary  

---

### 🔹 2. Exploratory Data Analysis (EDA)  
**Powered by Plotly (Interactive Visuals)**  

- 📊 Dropout Count (Bar Chart)  
- 🥧 Dropout Ratio (Pie Chart)  
- 👩‍🎓 Gender vs Dropout  
- 📚 Subjects vs Dropout  
- 📅 Semester vs Dropout  
- 🔥 Feature Correlation Heatmap  

---

### 🔹 3. Models & Performance Comparison  

Multiple models trained & evaluated:

- Logistic Regression  
- Random Forest  
- Gradient Boosting  

📈 **Model Evaluation Includes:**  
- Accuracy  
- Precision  
- Recall  
- F1-Score  
- Prediction Distribution Comparison  

✔ Accuracy alone is **NOT** the goal — **stability matters**

---

### 🔹 4. Real-World Dropout Risk Prediction  

Each student is assigned a **Dropout Risk Probability**:

| Risk Level | Meaning |
|-----------|--------|
| 🔴 High Risk | Immediate intervention required |
| 🟠 Medium Risk | Monitor closely |
| 🟢 Low Risk | Safe |

---

### 🔹 5. Key Insights  

- 🚨 **How many students can potentially drop out**
- 🔝 **Top 20 High-Risk Students**
- 🌟 **Top Safe / Best Students**
- 📊 Risk Distribution Visualization
- 🧠 **Top Reasons for Dropout (Feature Importance)**

---

### 🔹 6. Download Predictions  

📥 Download complete results including:
- Risk Score  
- Risk Label  
- Final Prediction  

---

## 🧪 Tech Stack  

- **Python**
- **Pandas, NumPy**
- **Scikit-Learn**
- **Plotly**
- **Streamlit**

---

## Hackathon Impact

✨ This system can be used by:

- Schools

- Colleges

- Universities

- EdTech Platforms

🎯 To reduce dropout rates, improve student retention, and enable data-driven educational policies.

---

## 👩‍💻 Author

- **Fozia Roshan**
- AI and Data Science | Machine Learning Enthusiast
- Hackathon Project 🚀

*⭐ If you like this project, don’t forget to star the repository!*

streamlit run dashboard.py
