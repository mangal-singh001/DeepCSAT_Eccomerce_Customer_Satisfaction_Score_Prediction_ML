## 🧠 **DeepCSAT: E-Commerce Customer Satisfaction Score Prediction (ML + DL Project)**

![Project Banner](https://img.shields.io/badge/Machine%20Learning-Ecommerce-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10+-yellow?style=for-the-badge\&logo=python)
![Framework](https://img.shields.io/badge/TensorFlow%20%7C%20Scikit--Learn-orange?style=for-the-badge\&logo=tensorflow)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

---

### 📌 **Project Overview**

The **DeepCSAT Project** aims to predict **Customer Satisfaction Scores (CSAT)** for an e-commerce platform using **Machine Learning and Deep Learning techniques**.
This project helps businesses **understand customer experience, predict satisfaction levels**, and **take proactive actions** to improve service quality and retention.

---

### 🎯 **Problem Statement**

In today’s competitive e-commerce landscape, retaining customers is as important as acquiring them.
Traditional satisfaction surveys are **reactive and delayed**, often capturing feedback only after dissatisfaction occurs.

Hence, the challenge is to build a **data-driven predictive model** that estimates **customer satisfaction** in real-time — based on features such as:

* Delivery performance
* Product quality
* Customer support
* Price perception
* Order accuracy, and more

This system empowers the business to identify at-risk customers early and make **strategic improvements** to enhance customer experience.

---

### 🎯 **Project Objective**

* Develop a predictive model that accurately estimates **Customer Satisfaction Score (CSAT)**.
* Identify key factors influencing satisfaction using **model explainability (SHAP)**.
* Deploy the final trained model using **Streamlit** for live, real-time predictions.

---

### 🧩 **Workflow / Project Pipeline**

1. **Data Collection & Cleaning**

   * Handling missing values, duplicates, and inconsistencies
2. **Exploratory Data Analysis (EDA)**

   * Statistical summaries and correlation heatmaps
3. **Feature Engineering**

   * Encoding categorical variables and scaling numerical features
4. **Model Development**

   * Linear Regression (Baseline)
   * Random Forest Regressor (Model-2)
   * Deep Learning ANN (Model-3)
5. **Model Evaluation**

   * Using MAE, MSE, RMSE, and R² metrics
6. **Explainability & Feature Importance**

   * SHAP analysis for business interpretability
7. **Model Saving & Deployment**

   * Joblib / TensorFlow SavedModel
   * Streamlit web app for real-time prediction

---

### 🤖 **Machine Learning Models Used**

| Model       | Technique               | Highlights                                             |
| :---------- | :---------------------- | :----------------------------------------------------- |
| **Model-1** | Linear Regression       | Baseline, interpretable but underfit                   |
| **Model-2** | Random Forest Regressor | Non-linear patterns, feature importance                |
| **Model-3** | Deep Learning (ANN)     | Captures complex feature interactions, best performing |

---

### 📈 **Evaluation Metrics**

| Metric                                | Meaning                           | Business Impact                                             |
| :------------------------------------ | :-------------------------------- | :---------------------------------------------------------- |
| **MAE (Mean Absolute Error)**         | Average absolute prediction error | Measures prediction reliability                             |
| **RMSE (Root Mean Squared Error)**    | Penalizes large errors            | Detects large satisfaction deviations                       |
| **R² (Coefficient of Determination)** | Explained variance                | Indicates how well the model captures satisfaction patterns |

---

### 🔍 **Explainability (SHAP Analysis)**

**SHAP (SHapley Additive exPlanations)** was used to interpret feature importance for the ANN model.

#### 💡 Top Influential Features:

* **Delivery Time ⬇️** – Longer deliveries reduce satisfaction
* **Product Quality ⬆️** – High-quality products increase satisfaction
* **Customer Support Score ⬆️** – Better support → happier customers
* **Order Accuracy ⬆️** – Fewer errors improve satisfaction

![SHAP Summary](https://github.com/mangal-singh001/DeepCSAT_Eccomerce_Customer_Satisfaction_Score_Prediction_ML/blob/main/images/shap_summary.png)

---

### 💾 **Model Saving and Deployment**

All final trained models and scalers were saved in the `artifacts/` folder:

```
📁 artifacts/
├── final_model_rf.joblib
├── final_ann_model/
├── final_scaler.joblib
```

The **Streamlit App** allows users to input customer details and instantly predict CSAT scores.

#### 🚀 Run Streamlit App

```bash
streamlit run app.py
```

---

### 🧮 **Tech Stack**

| Category                 | Tools / Libraries   |
| :----------------------- | :------------------ |
| **Programming Language** | Python              |
| **Data Handling**        | Pandas, NumPy       |
| **Visualization**        | Matplotlib, Seaborn |
| **Machine Learning**     | Scikit-learn        |
| **Deep Learning**        | TensorFlow / Keras  |
| **Model Explainability** | SHAP                |
| **Deployment**           | Streamlit           |
| **Serialization**        | Joblib / Pickle     |

---

### 🧾 **Results Summary**

| Dataset               |      MAE     |     RMSE     |       R²      |
| :-------------------- | :----------: | :----------: | :-----------: |
| **Linear Regression** |     1.05     |     1.37     |     0.008     |
| **Random Forest**     |     1.04     |     1.36     |     0.011     |
| **ANN (Final Model)** | **↓ Lowest** | **↓ Lowest** | **↑ Highest** |

✅ **Final Model Selected:** Deep Learning ANN
✅ **Reason:** Best trade-off between prediction accuracy and business interpretability.

---

### 🏁 **Conclusion**

The **DeepCSAT Model** successfully predicts e-commerce customer satisfaction by combining data analytics and deep learning.
It provides actionable insights into customer experience, enabling the company to:

* Identify dissatisfied customers early
* Improve delivery and support operations
* Increase customer retention and loyalty

This project demonstrates the power of **AI-driven customer experience management** in the e-commerce sector.

---

### 👨‍💻 **Developed By**

**👤 Mangal Singh**
📍 B.Tech – Computer Science & Engineering
🔗 [GitHub Profile](https://github.com/mangal-singh001)
💼 [LinkedIn](https://www.linkedin.com/in/mangal-singh001)

---

### ⭐ **Show Support**

If you like this project, please ⭐ the repo and share your feedback!
Your support helps me grow as a Data Scientist 💪

---

Would you like me to make a **shorter, visually styled version** of this README (with emojis, section dividers, and modern formatting) — perfect for **LinkedIn post or GitHub pinned README**?
