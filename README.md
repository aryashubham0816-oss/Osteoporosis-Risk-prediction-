# Osteoporosis Risk Prediction

An end-to-end machine learning pipeline for predicting osteoporosis risk using structured healthcare data. This project is designed for internship/project submission and demonstrates a complete ML workflow from data exploration to model deployment.

## Dataset Description

The dataset contains **1,958 patient records** with **15 features** (after removing ID):

| Feature | Type | Description |
|---------|------|-------------|
| Age | Numeric | Patient age (18-90 years) |
| Gender | Categorical | Male / Female |
| Hormonal Changes | Categorical | Normal / Postmenopausal |
| Family History | Categorical | Yes / No |
| Race/Ethnicity | Categorical | Asian / Caucasian / African American |
| Body Weight | Categorical | Underweight / Normal |
| Calcium Intake | Categorical | Low / Adequate |
| Vitamin D Intake | Categorical | Sufficient / Insufficient |
| Physical Activity | Categorical | Active / Sedentary |
| Smoking | Categorical | Yes / No |
| Alcohol Consumption | Categorical | Moderate |
| Medical Conditions | Categorical | Rheumatoid Arthritis / Hyperthyroidism |
| Medications | Categorical | Corticosteroids |
| Prior Fractures | Categorical | Yes / No |
| **Osteoporosis** | **Target** | **0 = No, 1 = Yes** |

The dataset is perfectly balanced (50% positive, 50% negative).

## Project Structure

```
C:\a_project1\
├── dataset\
│   └── raw_dataset_copy.csv       # Original dataset
├── reports\                       # Generated visualizations & results
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
├── main.py                        # Complete ML pipeline
├── trained_model.pkl              # Saved model + artifacts
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Installation

### Prerequisites
- Python 3.10+
- pip (Python package installer)

### Setup

```bash
# Clone the repository
git clone https://github.com/your-username/osteoporosis-prediction.git
cd osteoporosis-prediction

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn

## How to Run

```bash
python main.py
```

This will execute the complete pipeline:

1. **Load & Explore Data** — displays dataset overview, statistics, and info
2. **Exploratory Data Analysis** — generates all visualizations in `reports/`
3. **Data Preprocessing** — handles missing values, encodes categories, scales features
4. **Model Training** — trains Logistic Regression, Random Forest, SVM, and Decision Tree
5. **Hyperparameter Tuning** — GridSearchCV on Random Forest
6. **Model Evaluation** — confusion matrix, ROC curve, classification report
7. **Feature Importance** — identifies top risk factors
8. **Error Analysis** — analyzes false positives/negatives
9. **Save Artifacts** — saves trained model to `trained_model.pkl`

## Model Performance Summary

After hyperparameter tuning, the best model was **Random Forest** with:

| Metric | Value |
|--------|-------|
| Accuracy | 83.42% |
| Precision | 97.12% |
| Recall | 68.88% |
| F1 Score | 80.60% |
| ROC-AUC | 87.61% |

### Model Comparison (before tuning)

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Decision Tree | 83.42% | 95.17% | 70.41% | 80.94% | 86.82% |
| Random Forest | 83.16% | 93.33% | 71.43% | 80.92% | 87.90% |
| SVM | 81.89% | 91.39% | 70.41% | 79.54% | 87.65% |
| Logistic Regression | 79.85% | 84.21% | 73.47% | 78.47% | 87.49% |

### Best Hyperparameters (Random Forest)
```json
{
  "max_depth": 10,
  "min_samples_leaf": 4,
  "min_samples_split": 10,
  "n_estimators": 50
}
```

## Key Findings

1. **Age is the dominant risk factor** — accounts for ~83% of feature importance
2. **Race/Ethnicity, Family History, and Gender** are secondary contributors
3. **High precision (97%)** means very few false positives
4. **Moderate recall (69%)** — some at-risk patients are missed, highlighting the need for further improvement
5. **ROC-AUC of 0.88** indicates good overall discrimination ability

## Top Features for Prediction

1. Age
2. Race/Ethnicity
3. Family History
4. Gender
5. Calcium Intake
6. Prior Fractures
7. Hormonal Changes
8. Vitamin D Intake

## Error Analysis

- **False Positives: 4** — patients incorrectly flagged as high risk
- **False Negatives: 61** — at-risk patients not detected

Recall is especially important because missing a true osteoporosis case means a patient misses preventive treatment, potentially leading to fractures and reduced quality of life.

## Future Improvements

1. **Deep Learning** — use TensorFlow/Keras DNN for potentially better performance
2. **Class Imbalance Handling** — SMOTE or class weights if data becomes imbalanced
3. **More Features** — incorporate BMD (Bone Mineral Density) measurements, vitamin levels, hormone levels
4. **Explainability** — integrate SHAP/LIME for better model interpretability
5. **Deployment** — Streamlit web app or Flask API for real-time predictions
6. **Cross-Validation** — implement stratified k-fold for more robust evaluation
7. **Feature Engineering** — create interaction terms or polynomial features
8. **External Validation** — test on an independent dataset

## License

This project is for educational/demonstration purposes.

## Author

Submitted as part of internship/project work.
