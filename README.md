# 🦴 Osteoporosis Risk Prediction using Machine Learning

A complete Machine Learning project that predicts the risk of osteoporosis using patient demographic and clinical information. This project demonstrates an end-to-end ML pipeline including data preprocessing, exploratory data analysis (EDA), model training, hyperparameter tuning, evaluation, feature importance analysis, and model persistence.

---

## 📌 Project Overview

Osteoporosis is a common bone disease that weakens bones and increases fracture risk. Early identification of high-risk patients can help healthcare professionals recommend preventive treatments and lifestyle changes.

This project develops a predictive machine learning model using structured healthcare data to classify whether a patient is at risk of osteoporosis.

---

## ✨ Features

- Complete Exploratory Data Analysis (EDA)
- Data Cleaning & Preprocessing
- Feature Encoding
- Missing Value Analysis
- Correlation Analysis
- Multiple Machine Learning Models
- Hyperparameter Tuning using GridSearchCV
- Model Evaluation & Comparison
- Feature Importance Analysis
- ROC Curve & Precision-Recall Curve
- Model Serialization using Joblib

---

# 🛠 Tech Stack

| Category | Technologies |
|----------|--------------|
| Language | Python 3 |
| Libraries | Pandas, NumPy, Scikit-learn |
| Visualization | Matplotlib |
| Model Saving | Joblib |
| IDE | VS Code |

---

# 📂 Project Structure

```
osteoporosis-risk-prediction/
│
├── dataset/
│   └── raw_dataset_copy.csv
│
├── reports/
│   ├── age_analysis.png
│   ├── categorical_analysis.png
│   ├── confusion_matrix.png
│   ├── correlation_heatmap.png
│   ├── evaluation_results.csv
│   ├── feature_importance.png
│   ├── gender_analysis.png
│   ├── missing_values.png
│   ├── outlier_detection.png
│   ├── precision_recall_curve.png
│   ├── roc_curve.png
│   └── target_distribution.png
│
├── main.py
├── trained_model.pkl
├── requirements.txt
└── README.md
```

---

# 📊 Dataset Information

- **Total Records:** 1,958
- **Features:** 15
- **Target Variable:** Osteoporosis
- **Classes:** Balanced (50% Positive, 50% Negative)

### Features Used

- Age
- Gender
- Hormonal Changes
- Family History
- Race/Ethnicity
- Body Weight
- Calcium Intake
- Vitamin D Intake
- Physical Activity
- Smoking
- Alcohol Consumption
- Medical Conditions
- Medications
- Prior Fractures

Target:

- **0 → No Osteoporosis**
- **1 → Osteoporosis**

---

# ⚙ Machine Learning Workflow

```
Dataset
   │
   ▼
Data Cleaning
   │
   ▼
Exploratory Data Analysis
   │
   ▼
Feature Encoding
   │
   ▼
Train-Test Split
   │
   ▼
Model Training
   │
   ▼
Hyperparameter Tuning
   │
   ▼
Performance Evaluation
   │
   ▼
Feature Importance
   │
   ▼
Model Saving
```

---

# 🤖 Models Implemented

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)

---

# 🏆 Best Model

After hyperparameter tuning, the **Random Forest Classifier** achieved the best performance.

| Metric | Score |
|---------|-------|
| Accuracy | **83.42%** |
| Precision | **97.12%** |
| Recall | **68.88%** |
| F1 Score | **80.60%** |
| ROC-AUC | **87.61%** |

---

# 📈 Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score |
|--------|----------|-----------|--------|----------|
| Logistic Regression | 79.85% | 84.21% | 73.47% | 78.47% |
| Decision Tree | 83.42% | 95.17% | 70.41% | 80.94% |
| ⭐ Random Forest | **83.42%** | **97.12%** | **68.88%** | **80.60%** |
| SVM | 81.89% | 91.39% | 70.41% | 79.54% |

---

# 📌 Key Findings

- Age is the most influential predictor of osteoporosis.
- Family history significantly affects disease risk.
- Race/Ethnicity contributes to prediction performance.
- Calcium and Vitamin D intake improve prediction quality.
- Random Forest provided the most balanced performance among all models.

---

# 📊 Visualizations Generated

The project automatically generates:

- Target Distribution
- Age Distribution
- Gender Analysis
- Missing Value Analysis
- Correlation Heatmap
- Outlier Detection
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve
- Feature Importance Graph

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/aryashubham0816-oss/Osteoporosis-Risk-prediction-.git
```

Move into the project directory

```bash
cd Osteoporosis-Risk-prediction-
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶ Running the Project

```bash
python main.py
```

The program will automatically:

- Load the dataset
- Perform EDA
- Train all models
- Compare performance
- Tune Random Forest
- Generate visualizations
- Save the trained model

---

# 📌 Future Improvements

- Deploy as a Streamlit Web Application
- Build a Flask REST API
- Integrate SHAP for Explainable AI
- Include Bone Mineral Density (BMD) data
- Apply Deep Learning models
- Perform Cross-Validation
- Test on external clinical datasets

---

# 📜 License

This project is intended for educational and academic purposes.

---

# 👨‍💻 Author

**Arya pednekar**

Machine Learning | Python | Data Science

GitHub:
https://github.com/aryashubham0816-oss

---

⭐ If you found this project useful, consider giving it a Star.