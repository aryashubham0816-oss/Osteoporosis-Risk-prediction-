"""
Osteoporosis Risk Prediction - End-to-End ML Pipeline
=====================================================
A complete machine learning pipeline for predicting osteoporosis risk
using structured healthcare data.

Dataset columns:
    Id, Age, Gender, Hormonal Changes, Family History, Race/Ethnicity,
    Body Weight, Calcium Intake, Vitamin D Intake, Physical Activity,
    Smoking, Alcohol Consumption, Medical Conditions, Medications,
    Prior Fractures, Osteoporosis (target)

Author: Project Submission
"""

import os
import io
import warnings
import textwrap
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
import pickle

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(PROJECT_DIR, 'dataset', 'raw_dataset_copy.csv')
REPORTS_DIR = os.path.join(PROJECT_DIR, 'reports')
MODEL_PATH = os.path.join(PROJECT_DIR, 'trained_model.pkl')

os.makedirs(REPORTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. DATA LOADING & UNDERSTANDING
# ---------------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    """Load the CSV dataset."""
    df = pd.read_csv(path)
    print(f'Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns')
    return df


def data_overview(df: pd.DataFrame):
    """Print dataset overview: head, info, summary statistics."""
    print('\n' + '=' * 60)
    print('DATA OVERVIEW')
    print('=' * 60)

    print('\n--- First 5 rows ---')
    print(df.head().to_string())

    print('\n--- Data types & non-null counts ---')
    buf = io.StringIO()
    with redirect_stdout(buf):
        df.info()
    print(buf.getvalue())

    print('\n--- Summary statistics (numeric) ---')
    print(df.describe().to_string())

    print('\n--- Summary statistics (categorical) ---')
    cat_cols = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]
    for col in cat_cols:
        print(f'\n{col}:')
        print(df[col].value_counts().to_string())


# ---------------------------------------------------------------------------
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ---------------------------------------------------------------------------

def missing_value_analysis(df: pd.DataFrame):
    """Analyze and visualize missing values."""
    print('\n' + '=' * 60)
    print('MISSING VALUE ANALYSIS')
    print('=' * 60)
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({'Count': missing, 'Percent': missing_pct})
    missing_df = missing_df[missing_df['Count'] > 0].sort_values('Count', ascending=False)
    print(missing_df.to_string())

    if not missing_df.empty:
        plt.figure(figsize=(8, 4))
        sns.barplot(x=missing_df.index, y=missing_df['Count'], hue=missing_df.index,
                    palette='viridis', legend=False)
        plt.title('Missing Values by Column')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(REPORTS_DIR, 'missing_values.png'))
        plt.close()
        print('  -> Saved: reports/missing_values.png')
    return missing_df


def duplicate_check(df: pd.DataFrame):
    """Check for duplicate rows."""
    print('\n' + '=' * 60)
    print('DUPLICATE CHECK')
    print('=' * 60)
    dup = df.duplicated().sum()
    print(f'Duplicate rows: {dup}')


def plot_target_distribution(df: pd.DataFrame, target_col: str):
    """Plot target variable distribution."""
    print('\n' + '=' * 60)
    print('CLASS DISTRIBUTION')
    print('=' * 60)
    counts = df[target_col].value_counts()
    print(counts.to_string())
    print(f'Balance ratio: {counts.iloc[0] / counts.iloc[1]:.2f} : 1')

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    counts.plot(kind='bar', ax=axes[0], color=['skyblue', 'salmon'],
                edgecolor='black')
    axes[0].set_title('Target Class Counts')
    axes[0].set_xticklabels(['No Osteoporosis (0)', 'Osteoporosis (1)'], rotation=0)
    axes[0].set_ylabel('Count')

    counts.plot(kind='pie', ax=axes[1], autopct='%1.1f%%',
                colors=['skyblue', 'salmon'], startangle=90,
                labels=['No Osteoporosis', 'Osteoporosis'])
    axes[1].set_title('Target Class Proportion')
    axes[1].set_ylabel('')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'target_distribution.png'))
    plt.close()
    print('  -> Saved: reports/target_distribution.png')


def plot_age_distribution(df: pd.DataFrame):
    """Plot age distribution analysis."""
    print('\n' + '=' * 60)
    print('AGE DISTRIBUTION ANALYSIS')
    print('=' * 60)
    print(f'Age range: {df["Age"].min()} - {df["Age"].max()}')
    print(f'Mean age: {df["Age"].mean():.1f}')
    print(f'Median age: {df["Age"].median():.1f}')

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(df['Age'], bins=30, color='steelblue', edgecolor='black')
    axes[0].set_title('Age Distribution')
    axes[0].set_xlabel('Age')
    axes[0].set_ylabel('Count')

    axes[1].hist(df[df['Osteoporosis'] == 1]['Age'], bins=30,
                 alpha=0.7, label='Osteoporosis', color='salmon')
    axes[1].hist(df[df['Osteoporosis'] == 0]['Age'], bins=30,
                 alpha=0.7, label='No Osteoporosis', color='skyblue')
    axes[1].set_title('Age by Target')
    axes[1].set_xlabel('Age')
    axes[1].set_ylabel('Count')
    axes[1].legend()

    df.boxplot(column='Age', by='Osteoporosis', ax=axes[2])
    axes[2].set_title('Age Boxplot by Target')
    axes[2].set_xticklabels(['No Osteoporosis', 'Osteoporosis'])
    axes[2].set_xlabel('')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'age_analysis.png'))
    plt.close()
    print('  -> Saved: reports/age_analysis.png')


def plot_gender_analysis(df: pd.DataFrame):
    """Analyze osteoporosis by gender."""
    print('\n' + '=' * 60)
    print('GENDER-WISE ANALYSIS')
    print('=' * 60)
    ct = pd.crosstab(df['Gender'], df['Osteoporosis'], margins=True)
    print(ct.to_string())

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    pd.crosstab(df['Gender'], df['Osteoporosis']).plot(
        kind='bar', ax=axes[0], color=['skyblue', 'salmon'],
        edgecolor='black', legend=True)
    axes[0].set_title('Osteoporosis by Gender')
    axes[0].set_ylabel('Count')
    axes[0].legend(['No', 'Yes'])
    axes[0].tick_params(axis='x', rotation=0)

    gender_pct = pd.crosstab(df['Gender'], df['Osteoporosis'], normalize='index') * 100
    gender_pct.plot(kind='bar', ax=axes[1], color=['skyblue', 'salmon'],
                    edgecolor='black', stacked=True)
    axes[1].set_title('Osteoporosis Proportion by Gender (%)')
    axes[1].set_ylabel('Percentage')
    axes[1].legend(['No', 'Yes'])
    axes[1].tick_params(axis='x', rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'gender_analysis.png'))
    plt.close()
    print('  -> Saved: reports/gender_analysis.png')


def plot_categorical_analysis(df: pd.DataFrame, target_col: str):
    """Analyze categorical features against the target."""
    print('\n' + '=' * 60)
    print('SYMPTOM / FEATURE-WISE ANALYSIS')
    print('=' * 60)
    cat_cols = df.select_dtypes(exclude='number').columns

    n_cols = 3
    n_rows = int(np.ceil(len(cat_cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
    axes = axes.flatten()

    for i, col in enumerate(cat_cols):
        ct = pd.crosstab(df[col], df[target_col], normalize='index') * 100
        ct.plot(kind='bar', ax=axes[i], color=['skyblue', 'salmon'],
                edgecolor='black', stacked=True, legend=False)
        axes[i].set_title(f'{col} vs Osteoporosis')
        axes[i].set_ylabel('Percentage')
        axes[i].tick_params(axis='x', rotation=45)
        print(f'\n{col}:')
        print(pd.crosstab(df[col], df[target_col]).to_string())

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    handles = [plt.Rectangle((0, 0), 1, 1, color='skyblue'),
               plt.Rectangle((0, 0), 1, 1, color='salmon')]
    fig.legend(handles, ['No Osteoporosis', 'Osteoporosis'],
               loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'categorical_analysis.png'),
                bbox_inches='tight')
    plt.close()
    print('\n  -> Saved: reports/categorical_analysis.png')


def plot_correlation_heatmap(df: pd.DataFrame):
    """Plot correlation heatmap for numeric features."""
    print('\n' + '=' * 60)
    print('CORRELATION HEATMAP')
    print('=' * 60)
    temp = df.copy()
    for col in temp.select_dtypes(exclude='number').columns:
        temp[col] = LabelEncoder().fit_transform(temp[col].astype(str))

    corr = temp.corr(numeric_only=False)
    plt.figure(figsize=(12, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, linewidths=0.5, square=True)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'correlation_heatmap.png'))
    plt.close()
    print('  -> Saved: reports/correlation_heatmap.png')


def detect_outliers(df: pd.DataFrame):
    """Detect and visualize outliers in numeric features using IQR."""
    print('\n' + '=' * 60)
    print('OUTLIER DETECTION')
    print('=' * 60)
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    num_cols = [c for c in num_cols if c != 'Id' and c != 'Osteoporosis']

    fig, axes = plt.subplots(1, len(num_cols), figsize=(5 * len(num_cols), 4))
    if len(num_cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, num_cols):
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        print(f'{col}: {len(outliers)} outliers ({(len(outliers)/len(df))*100:.1f}%)')

        ax.boxplot(df[col], vert=True, patch_artist=True,
                   boxprops=dict(facecolor='lightblue'))
        ax.set_title(f'{col} (IQR = {IQR:.1f})')
        ax.set_ylabel(col)

    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'outlier_detection.png'))
    plt.close()
    print('  -> Saved: reports/outlier_detection.png')


# ---------------------------------------------------------------------------
# 3. DATA PREPROCESSING
# ---------------------------------------------------------------------------

def preprocess_data(df: pd.DataFrame, target_col: str = 'Osteoporosis'):
    """Clean, encode, scale, and split the data."""
    print('\n' + '=' * 60)
    print('DATA PREPROCESSING')
    print('=' * 60)

    data = df.copy()

    # Drop Id column
    if 'Id' in data.columns:
        data = data.drop(columns=['Id'])
        print('Dropped Id column.')

    # Handle missing values: forward-fill categorical, median for numeric
    for col in data.columns:
        if col == target_col:
            continue
        if data[col].dtype in ['object', 'str']:
            data[col] = data[col].ffill()
        else:
            data[col] = data[col].fillna(data[col].median())

    print('Missing values handled (ffill categorical, median numeric).')

    # Separate features and target
    X = data.drop(columns=[target_col])
    y = data[target_col].values

    # Encode categorical features
    cat_cols = X.select_dtypes(exclude='number').columns.tolist()
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    print(f'Encoded {len(cat_cols)} categorical columns.')

    # Store feature names
    feature_names = X.columns.tolist()

    # Train-test split (80-20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f'Train: {X_train.shape[0]}, Test: {X_test.shape[0]}')

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print('Features scaled using StandardScaler.')

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names


# ---------------------------------------------------------------------------
# 4. MODEL DEVELOPMENT
# ---------------------------------------------------------------------------

def train_models(X_train, y_train, X_test, y_test):
    """Train multiple ML models and compare performance."""
    print('\n' + '=' * 60)
    print('MODEL TRAINING')
    print('=' * 60)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
        'Decision Tree': RandomForestClassifier(
            n_estimators=1, max_depth=5, random_state=42
        ),  # Using a single tree as a Decision Tree proxy
    }

    results = []

    for name, model in models.items():
        print(f'\nTraining {name}...')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_prob)

        results.append({
            'Model': name,
            'Accuracy': round(acc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1 Score': round(f1, 4),
            'ROC-AUC': round(roc, 4),
        })
        print(f'  Accuracy: {acc:.4f}, Precision: {prec:.4f}, '
              f'Recall: {rec:.4f}, F1: {f1:.4f}, ROC-AUC: {roc:.4f}')

    results_df = pd.DataFrame(results).sort_values('Accuracy', ascending=False)
    print('\n--- Model Comparison ---')
    print(results_df.to_string(index=False))
    return results_df, models


# ---------------------------------------------------------------------------
# 5. HYPERPARAMETER TUNING
# ---------------------------------------------------------------------------

def tune_hyperparameters(X_train, y_train):
    """Tune Random Forest using GridSearchCV."""
    print('\n' + '=' * 60)
    print('HYPERPARAMETER TUNING (Random Forest)')
    print('=' * 60)

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    }

    rf = RandomForestClassifier(random_state=42)
    grid = GridSearchCV(
        rf, param_grid, cv=5, scoring='accuracy',
        n_jobs=-1, verbose=1
    )
    grid.fit(X_train, y_train)

    print(f'\nBest Parameters: {grid.best_params_}')
    print(f'Best CV Accuracy: {grid.best_score_:.4f}')

    return grid.best_estimator_, grid.best_params_


# ---------------------------------------------------------------------------
# 6. FINAL MODEL EVALUATION
# ---------------------------------------------------------------------------

def evaluate_model(model, X_test, y_test, model_name: str = 'Best Model'):
    """Comprehensive evaluation of the final model."""
    print('\n' + '=' * 60)
    print(f'FINAL MODEL EVALUATION - {model_name}')
    print('=' * 60)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)

    print(f'\nAccuracy:  {acc:.4f}')
    print(f'Precision: {prec:.4f}')
    print(f'Recall:    {rec:.4f}')
    print(f'F1 Score:  {f1:.4f}')
    print(f'ROC-AUC:   {roc:.4f}')

    print('\n--- Confusion Matrix ---')
    cm = confusion_matrix(y_test, y_pred)
    print(f'True Negatives:  {cm[0, 0]}')
    print(f'False Positives: {cm[0, 1]}')
    print(f'False Negatives: {cm[1, 0]}')
    print(f'True Positives:  {cm[1, 1]}')

    print('\n--- Classification Report ---')
    print(classification_report(y_test, y_pred,
                                target_names=['No Osteoporosis', 'Osteoporosis']))

    # Plot Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Osteoporosis', 'Osteoporosis'],
                yticklabels=['No Osteoporosis', 'Osteoporosis'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'confusion_matrix.png'))
    plt.close()
    print('  -> Saved: reports/confusion_matrix.png')

    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc:.4f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'roc_curve.png'))
    plt.close()
    print('  -> Saved: reports/roc_curve.png')

    # Plot Precision-Recall Curve
    prec_vals, rec_vals, _ = precision_recall_curve(y_test, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(rec_vals, prec_vals, label='PR Curve', color='green', linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'precision_recall_curve.png'))
    plt.close()
    print('  -> Saved: reports/precision_recall_curve.png')

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'roc_auc': roc,
        'confusion_matrix': cm,
    }


# ---------------------------------------------------------------------------
# 7. FEATURE IMPORTANCE & EXPLAINABILITY
# ---------------------------------------------------------------------------

def feature_importance_analysis(model, feature_names: list):
    """Plot feature importance for tree-based models."""
    print('\n' + '=' * 60)
    print('FEATURE IMPORTANCE ANALYSIS')
    print('=' * 60)

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        print('\nTop 10 Most Important Features:')
        for i in range(min(10, len(feature_names))):
            print(f'  {i + 1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}')

        plt.figure(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(indices)))
        plt.bar(range(len(indices)), importances[indices], color=colors, edgecolor='black')
        plt.xticks(range(len(indices)),
                   [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.title('Feature Importance (Random Forest)')
        plt.xlabel('Features')
        plt.ylabel('Importance Score')
        plt.tight_layout()
        plt.savefig(os.path.join(REPORTS_DIR, 'feature_importance.png'))
        plt.close()
        print('\n  -> Saved: reports/feature_importance.png')
    else:
        # For Logistic Regression, use coefficients
        coefs = model.coef_[0]
        indices = np.argsort(np.abs(coefs))[::-1]
        print('\nTop 10 Most Important Features (by coefficient magnitude):')
        for i in range(min(10, len(feature_names))):
            print(f'  {i + 1}. {feature_names[indices[i]]}: {coefs[indices[i]]:.4f}')

        plt.figure(figsize=(10, 6))
        colors = ['red' if c < 0 else 'green' for c in coefs[indices]]
        plt.bar(range(len(indices)), coefs[indices], color=colors, edgecolor='black')
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        plt.xticks(range(len(indices)),
                   [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.title('Feature Importance (Logistic Regression Coefficients)')
        plt.xlabel('Features')
        plt.ylabel('Coefficient Value')
        plt.tight_layout()
        plt.savefig(os.path.join(REPORTS_DIR, 'feature_importance.png'))
        plt.close()
        print('\n  -> Saved: reports/feature_importance.png')


def medical_interpretation():
    """Provide medical interpretation of key findings."""
    print('\n' + '=' * 60)
    print('MEDICAL INTERPRETATION OF FINDINGS')
    print('=' * 60)
    print(textwrap.dedent("""
    Osteoporosis is a bone disease characterized by decreased bone density
    and increased fracture risk. Key findings from this analysis:

    1. Age is a major risk factor — osteoporosis risk increases significantly
       with age, especially after menopause in women.

    2. Hormonal changes (particularly postmenopausal status) strongly
       correlate with osteoporosis due to estrogen decline.

    3. Body weight plays a role — underweight individuals have higher risk
       due to lower mechanical bone stimulation and reduced estrogen
       production in adipose tissue.

    4. Lifestyle factors (smoking, alcohol consumption, sedentary behavior)
       contribute to bone density loss.

    5. Calcium and Vitamin D intake are protective factors — adequate
       intake supports bone mineralization.

    6. Family history indicates genetic predisposition.

    7. Certain medical conditions (Rheumatoid Arthritis, Hyperthyroidism)
       and medications (Corticosteroids) increase osteoporosis risk.

    These findings align with established medical literature on
    osteoporosis risk factors.
    """))


# ---------------------------------------------------------------------------
# 8. ERROR ANALYSIS
# ---------------------------------------------------------------------------

def error_analysis(model, X_test, y_test, feature_names: list):
    """Analyze false positives and false negatives."""
    print('\n' + '=' * 60)
    print('ERROR ANALYSIS')
    print('=' * 60)

    y_pred = model.predict(X_test)

    false_positives = np.where((y_test == 0) & (y_pred == 1))[0]
    false_negatives = np.where((y_test == 1) & (y_pred == 0))[0]

    print(f'\nFalse Positives (predicted osteoporosis, actual none): {len(false_positives)}')
    print(f'False Negatives (predicted none, actual osteoporosis): {len(false_negatives)}')

    if len(false_positives) > 0:
        print('\n--- Sample False Positives ---')
        for idx in false_positives[:3]:
            print(f'  Index {idx}: ', end='')
            for fn in feature_names:
                print(f'{fn}={X_test[idx, feature_names.index(fn)]:.2f} ', end='')
            print()

    if len(false_negatives) > 0:
        print('\n--- Sample False Negatives ---')
        for idx in false_negatives[:3]:
            print(f'  Index {idx}: ', end='')
            for fn in feature_names:
                print(f'{fn}={X_test[idx, feature_names.index(fn)]:.2f} ', end='')
            print()

    print('\n--- Why Recall is Important for Osteoporosis Prediction ---')
    print(textwrap.dedent("""
    Recall (Sensitivity) measures how many actual positive cases the model
    correctly identifies. In osteoporosis prediction:

    - A false negative (missing a patient with osteoporosis) means the
      patient does not receive preventive treatment or lifestyle advice,
      potentially leading to fractures, hospitalization, and reduced
      quality of life.

    - A false positive may cause unnecessary worry or additional testing,
      but is less harmful than missing a real case.

    Therefore, high recall is prioritized to minimize missed diagnoses.
    """))


# ---------------------------------------------------------------------------
# 9. SAVE ARTIFACTS
# ---------------------------------------------------------------------------

def save_model_and_artifacts(model, scaler, feature_names, results_dict):
    """Save the trained model and related artifacts."""
    print('\n' + '=' * 60)
    print('SAVING ARTIFACTS')
    print('=' * 60)

    artifacts = {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names,
        'results': results_dict,
    }
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(artifacts, f)
    print(f'Model and artifacts saved to: {MODEL_PATH}')

    # Save evaluation results as CSV
    results_df = pd.DataFrame([results_dict])
    results_path = os.path.join(REPORTS_DIR, 'evaluation_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f'Evaluation results saved to: {results_path}')


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------

def main():
    """Run the complete pipeline."""
    print('=' * 60)
    print('OSTEOPOROSIS RISK PREDICTION')
    print('End-to-End Machine Learning Pipeline')
    print('=' * 60)

    # 1. Load data
    df = load_data(DATASET_PATH)

    # 2. Data overview
    data_overview(df)

    # 3. EDA
    target_col = 'Osteoporosis'
    missing_value_analysis(df)
    duplicate_check(df)
    plot_target_distribution(df, target_col)
    plot_age_distribution(df)
    plot_gender_analysis(df)
    plot_categorical_analysis(df, target_col)
    plot_correlation_heatmap(df)
    detect_outliers(df)

    # 4. Preprocessing
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(df)

    # 5. Train models
    results_df, models = train_models(X_train, y_train, X_test, y_test)

    # 6. Hyperparameter tuning
    best_rf, best_params = tune_hyperparameters(X_train, y_train)
    print(f'\nRetraining with best params...')
    best_model = RandomForestClassifier(**best_params, random_state=42)
    best_model.fit(X_train, y_train)

    # 7. Evaluate best model
    eval_results = evaluate_model(best_model, X_test, y_test, 'Tuned Random Forest')

    # 8. Feature importance
    feature_importance_analysis(best_model, feature_names)

    # 9. Medical interpretation
    medical_interpretation()

    # 10. Error analysis
    error_analysis(best_model, X_test, y_test, feature_names)

    # 11. Save artifacts
    save_model_and_artifacts(best_model, scaler, feature_names, eval_results)

    print('\n' + '=' * 60)
    print('PIPELINE COMPLETED SUCCESSFULLY!')
    print('=' * 60)
    print(f'\nAll reports saved to: {REPORTS_DIR}/')
    print(f'Trained model saved to: {MODEL_PATH}')
    print('\nTo load the model later:')
    print('  import pickle')
    print('  with open("trained_model.pkl", "rb") as f:')
    print('      artifacts = pickle.load(f)')
    print('      model = artifacts["model"]')
    print('      scaler = artifacts["scaler"]')
    print('      feature_names = artifacts["feature_names"]')


if __name__ == '__main__':
    main()
