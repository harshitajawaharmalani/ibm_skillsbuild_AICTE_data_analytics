#!/usr/bin/env python
# coding: utf-8

# # IBM SkillsBuild Data Analytics with AI Academic Internship
# ### Conducted by BharatCares in association with AICTE
# 
# ---
# # 📊 Telco Customer Churn Analytics and Predictive Modeling Using AI 
# 
# ---
# ### 🎯 Project Overview & Objective
# Customer churn represents one of the most critical challenges facing the modern telecommunications industry. Acquiring a new subscriber is estimated to cost between 5 to 7 times more than retaining an existing customer. 
# 
# The primary objectives of this project are:
# 1. **In-depth Exploratory Data Analysis (EDA):** Uncover the demographic, contractual, and service usage patterns directly driving customer attrition.
# 2. **Data Cleaning & Engineering:** Standardize messy data, resolve data type mismatches, and prepare robust feature pipelines.
# 3. **AI & Machine Learning Modeling:** Implement, evaluate, and compare multiple supervised machine learning architectures (Logistic Regression, Random Forest, and Gradient Boosting).
# 4. **Model Evaluation & Explainability:** Evaluate models using Accuracy, Precision, Recall, F1-Score, and ROC-AUC curves, identifying the most predictive churn indicators.
# 5. **Prescriptive Business Strategy:** Translate model insights into actionable retention interventions for telecom executives.
# 

# In[1]:


# ==========================================
# 1. SETUP & LIBRARY IMPORTS
# ==========================================
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-Learn Modules
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

# Plotting Configuration
sns.set_theme(style='whitegrid')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['figure.titlesize'] = 14

print('✓ All core libraries imported successfully!')


# ## 1. Data Ingestion & Preliminary Inspection
# We load the authentic IBM Telco Customer Churn dataset, inspecting its dimensionality, column types, and first few records.
# 

# In[2]:


# Load the dataset
data_file = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
df = pd.read_csv(data_file)

print(f'Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns')
print('=' * 60)
print('Dataset Column List:')
print(list(df.columns))
print('=' * 60)
df.head()


# In[3]:


# Inspect data types and non-null counts
df.info()


# ## 2. Data Cleaning & Type Casting
# ### Key Data Observations & Resolutions:
# - **TotalCharges Data Type Mismatch:** `TotalCharges` was loaded as an `object` (string) because 11 rows contained whitespace characters (`" "`).
# - **Zero Tenure Attribution:** These 11 records correspond strictly to newly joined subscribers where `tenure == 0`.
# - **Numerical Type Conversion:** Converted `TotalCharges` to `float64` using `pd.to_numeric` with coercion, imputing missing charges with `0.0`.
# - **Identifier Removal:** `customerID` is an arbitrary hexadecimal customer key providing zero predictive value, hence removed.
# - **Target Variable Mapping:** The binary response variable `Churn` contains `'No'` and `'Yes'`, which is mapped to integer values `0` and `1`.
# 

# In[4]:


# Clean TotalCharges and handle whitespace anomalies
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(' ', np.nan), errors='coerce')
missing_total_charges = df['TotalCharges'].isna().sum()
print(f'Number of missing TotalCharges records imputed: {missing_total_charges}')

# Impute with 0.0 (since tenure is 0 for these accounts)
df['TotalCharges'] = df['TotalCharges'].fillna(0.0)

# Drop non-predictive identifier
df_clean = df.drop(columns=['customerID']).copy()

# Summary statistics of numerical columns
df_clean.describe().T


# ## 3. Exploratory Data Analysis (EDA) — Overall Churn Distribution
# Understanding class imbalance is vital for choosing appropriate evaluation metrics and loss functions.
# 

# In[5]:


# Churn class distribution
churn_counts = df_clean['Churn'].value_counts()
churn_pct = df_clean['Churn'].value_counts(normalize=True) * 100

print('Churn Class Distribution:')
for status, count in churn_counts.items():
    print(f' - {status}: {count:,} ({churn_pct[status]:.2f}%)')

plt.figure(figsize=(7, 5))
colors = ['#2b5c8f', '#d9534f']
bars = plt.bar(churn_counts.index, churn_counts.values, color=colors, width=0.5, edgecolor='black', alpha=0.85)
for bar in bars:
    yval = bar.get_height()
    pct = (yval / len(df_clean)) * 100
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 60, f'{yval:,} ({pct:.1f}%)', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.title('Overall Customer Churn Distribution (IBM Telco Dataset)', fontsize=13, fontweight='bold', pad=15)
plt.xlabel('Churn Status', fontsize=11, fontweight='bold')
plt.ylabel('Number of Subscribers', fontsize=11, fontweight='bold')
plt.ylim(0, max(churn_counts.values) * 1.15)
plt.tight_layout()
plt.show()


# ## 4. Demographic Analysis vs Customer Churn
# We explore demographic characteristics: `gender`, `SeniorCitizen`, `Partner`, and `Dependents`.
# 

# In[6]:


demo_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents']
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
axes = axes.flatten()

for i, col in enumerate(demo_cols):
    ct = pd.crosstab(df_clean[col], df_clean['Churn'], normalize='index') * 100
    ct.plot(kind='bar', stacked=True, ax=axes[i], color=['#2b5c8f', '#d9534f'], edgecolor='black', alpha=0.85)
    axes[i].set_title(f'Churn Breakdown by {col}', fontsize=12, fontweight='bold')
    axes[i].set_ylabel('Percentage (%)', fontsize=10)
    axes[i].set_xlabel(col, fontsize=10)
    axes[i].legend(['Retained (No)', 'Churned (Yes)'], loc='upper right')
    axes[i].tick_params(axis='x', rotation=0)

plt.suptitle('Demographic Drivers of Customer Churn', fontsize=15, fontweight='bold', y=0.99)
plt.tight_layout()
plt.show()


# ## 5. Contract Dynamics and Billing Method Analysis
# Contracts and payment mechanics represent two of the strongest operational levers for telecom providers. Let us inspect their empirical churn rates.
# 

# In[7]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

ct_contract = pd.crosstab(df_clean['Contract'], df_clean['Churn'], normalize='index') * 100
ct_contract.plot(kind='bar', ax=ax1, color=['#2b5c8f', '#d9534f'], edgecolor='black', alpha=0.85)
ax1.set_title('Churn Rate by Contract Duration', fontsize=12, fontweight='bold')
ax1.set_ylabel('Percentage (%)', fontsize=10)
ax1.set_xlabel('Contract Type', fontsize=10)
ax1.legend(['Retained', 'Churned'])
ax1.tick_params(axis='x', rotation=15)

ct_payment = pd.crosstab(df_clean['PaymentMethod'], df_clean['Churn'], normalize='index') * 100
ct_payment.plot(kind='bar', ax=ax2, color=['#2b5c8f', '#d9534f'], edgecolor='black', alpha=0.85)
ax2.set_title('Churn Rate by Payment Mechanism', fontsize=12, fontweight='bold')
ax2.set_ylabel('Percentage (%)', fontsize=10)
ax2.set_xlabel('Payment Method', fontsize=10)
ax2.legend(['Retained', 'Churned'])
ax2.tick_params(axis='x', rotation=25)

plt.suptitle('Contract & Payment Analysis', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.show()


# ## 6. Subscribed Services & Value-Add Add-ons
# How does the presence of technical support, online security, and internet service tier influence customer retention?
# 

# In[8]:


services = ['InternetService', 'OnlineSecurity', 'TechSupport', 'PaperlessBilling']
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, col in enumerate(services):
    ct = pd.crosstab(df_clean[col], df_clean['Churn'], normalize='index') * 100
    ct.plot(kind='bar', ax=axes[i], color=['#2b5c8f', '#d9534f'], edgecolor='black', alpha=0.85)
    axes[i].set_title(f'Churn Rate by {col}', fontsize=12, fontweight='bold')
    axes[i].set_ylabel('Percentage (%)', fontsize=10)
    axes[i].tick_params(axis='x', rotation=15)
    axes[i].legend(['Retained', 'Churned'], loc='upper right')

plt.suptitle('Service Offerings & Billing Preference vs Churn', fontsize=15, fontweight='bold', y=0.99)
plt.tight_layout()
plt.show()


# ## 7. Numerical Distributions: Tenure & Financial Charges
# Let us examine the kernel density estimates of `tenure`, `MonthlyCharges`, and `TotalCharges` across retained and churned cohorts.
# 

# In[9]:


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

sns.kdeplot(data=df_clean, x='tenure', hue='Churn', fill=True, common_norm=False, palette=['#2b5c8f', '#d9534f'], ax=ax1, alpha=0.4)
ax1.set_title('Tenure Distribution (Months)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Tenure (Months)', fontsize=10)

sns.kdeplot(data=df_clean, x='MonthlyCharges', hue='Churn', fill=True, common_norm=False, palette=['#2b5c8f', '#d9534f'], ax=ax2, alpha=0.4)
ax2.set_title('Monthly Charges Distribution ($)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Monthly Charges ($)', fontsize=10)

sns.kdeplot(data=df_clean, x='TotalCharges', hue='Churn', fill=True, common_norm=False, palette=['#2b5c8f', '#d9534f'], ax=ax3, alpha=0.4)
ax3.set_title('Total Charges Distribution ($)', fontsize=12, fontweight='bold')
ax3.set_xlabel('Total Charges ($)', fontsize=10)

plt.suptitle('Distribution of Continuous Metrics by Churn Status', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


# ## 8. Correlation Analysis
# We map the target variable to binary integer format (`0` = No, `1` = Yes) and calculate Pearson correlation coefficients.
# 

# In[10]:


df_encoded_temp = df_clean.copy()
df_encoded_temp['Churn_Binary'] = df_encoded_temp['Churn'].map({'No': 0, 'Yes': 1})
num_corr = df_encoded_temp[['Churn_Binary', 'tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(num_corr, annot=True, cmap='Blues', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Numerical Features & Churn', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()


# ## 9. Data Preprocessing & Machine Learning Pipeline
# 1. **Target Encoding:** Map `Churn` to `0` and `1`.
# 2. **One-Hot Encoding:** Convert categorical attributes using `pd.get_dummies(..., drop_first=True)` to prevent multicollinearity.
# 3. **Train-Test Split:** Perform an 80/20 stratified split to preserve class ratios.
# 4. **Feature Normalization:** Apply `StandardScaler` to continuous numerical columns (`tenure`, `MonthlyCharges`, `TotalCharges`).
# 

# In[11]:


# Encode target
df_clean['Churn'] = df_clean['Churn'].map({'No': 0, 'Yes': 1})

# One-hot encoding
df_final = pd.get_dummies(df_clean, drop_first=True)
print(f'Encoded Feature Matrix Shape: {df_final.shape[0]} rows, {df_final.shape[1]} columns')

X = df_final.drop(columns=['Churn'])
y = df_final['Churn']

# Stratified Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standard Scaling
scaler = StandardScaler()
num_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[num_features] = scaler.fit_transform(X_train[num_features])
X_test_scaled[num_features] = scaler.transform(X_test[num_features])

print(f'Training Set: {X_train.shape[0]} samples')
print(f'Testing Set:  {X_test.shape[0]} samples')


# ## 10. Artificial Intelligence & Machine Learning Modeling
# We train three complementary machine learning architectures:
# 1. **Logistic Regression:** Linear probabilistic classifier providing interpretable odds ratios.
# 2. **Random Forest Classifier:** Bagging ensemble composed of decorrelated decision trees.
# 3. **Gradient Boosting Classifier:** Sequential boosting ensemble optimizing residual errors.
# 

# In[12]:


# Initialize models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=150, learning_rate=0.08, max_depth=4, random_state=42)
}

results = {}
predictions = {}
predicted_probabilities = {}

for name, model in models.items():
    print(f'Training {name}...')
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    results[name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_prob)
    }
    predictions[name] = y_pred
    predicted_probabilities[name] = y_prob
    print(f'✓ {name} completed successfully.')


# ## 11. Confusion Matrix Evaluation
# Evaluating False Positives (retaining customers who would not have left) and False Negatives (missing at-risk customers).
# 

# In[13]:


fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for i, (name, y_pred) in enumerate(predictions.items()):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[i],
                xticklabels=['Retained (0)', 'Churned (1)'],
                yticklabels=['Retained (0)', 'Churned (1)'])
    axes[i].set_title(f'{name} - Confusion Matrix', fontsize=12, fontweight='bold')
    axes[i].set_xlabel('Predicted Class', fontsize=10)
    axes[i].set_ylabel('True Class', fontsize=10)

plt.tight_layout()
plt.show()


# ## 12. Model Performance Benchmark Comparison
# A comprehensive side-by-side comparison across all key classification metrics.
# 

# In[14]:


benchmark_df = pd.DataFrame(results).T
benchmark_df = benchmark_df.round(4)
print('Model Evaluation Benchmark:')
print(benchmark_df)


# ## 13. Receiver Operating Characteristic (ROC) & AUC Curves
# Comparing the discrimination threshold performance across all three algorithms.
# 

# In[15]:


plt.figure(figsize=(8, 6))

for name, y_prob in predicted_probabilities.items():
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_val = results[name]['ROC-AUC']
    plt.plot(fpr, tpr, lw=2.2, label=f'{name} (ROC-AUC = {auc_val:.3f})')

plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Baseline (AUC = 0.500)')
plt.title('ROC-AUC Curves Comparison', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11, fontweight='bold')
plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=11, fontweight='bold')
plt.legend(loc='lower right', fontsize=11)
plt.tight_layout()
plt.show()


# ## 14. Explainable AI (XAI): Feature Importance
# Understanding which business indicators have the highest predictive weight in classifying churn risk using our Gradient Boosting ensemble.
# 

# In[16]:


gb_clf = models['Gradient Boosting']
feat_importances = pd.Series(gb_clf.feature_importances_, index=X.columns)
top15_features = feat_importances.sort_values(ascending=False).head(15)

plt.figure(figsize=(10, 6))
top15_features.sort_values().plot(kind='barh', color='#2b5c8f', edgecolor='black', alpha=0.85)
plt.title('Top 15 Most Influential Features Predicting Customer Churn (Gradient Boosting)', fontsize=12, fontweight='bold')
plt.xlabel('Relative Feature Importance Score', fontsize=11, fontweight='bold')
plt.ylabel('Feature', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.show()


# ## 15. Strategic Business Recommendations
# Based on the empirical evidence and AI modeling findings:
# 1. **Contract Migration Incentives:** Customers on month-to-month contracts exhibit over **42% churn rate**, compared to **<11%** for 1-year and **<3%** for 2-year contracts. Offer introductory 1-year loyalty discounts.
# 2. **First-Year Onboarding Program:** Churn is heavily concentrated in the first **12 months of tenure**. Establish a proactive customer success check-in at 30, 90, and 180 days.
# 3. **Fiber Optic Service Quality & Pricing Review:** Fiber optic subscribers experience disproportionate churn rates (~41.9%) despite paying premium rates, indicating potential network instability or perceived cost-to-value mismatch.
# 4. **Bundle Tech Support & Online Security:** Customers with Tech Support and Online Security have substantially lower attrition rates (~15% vs ~41%). Bundle these services for free during initial onboarding.
# 5. **Promote Automated Bank/Credit Card Payments:** Electronic check users exhibit an alarming **45% churn rate** versus **~16%** for automated bank transfers. Provide automated payment sign-up rebates.
# 

# In[17]:


# High-Risk Segment Identification & Targeted Campaign Sizing
high_risk_mask = (y_test == 1) & (predicted_probabilities['Gradient Boosting'] >= 0.70)
num_flagged = high_risk_mask.sum()
avg_monthly_rev = X_test.loc[high_risk_mask, 'MonthlyCharges'].mean()
annual_rev_at_risk = num_flagged * avg_monthly_rev * 12

print('=' * 60)
print('🎯 AI-DRIVEN RETENTION CAMPAIGN SIZING')
print('=' * 60)
print(f'Test Cohort High-Confidence Churn Identified: {num_flagged} subscribers')
print(f'Average Monthly Spend per High-Risk Subscriber: ${avg_monthly_rev:.2f}')
print(f'Annual Revenue Protected via Early Intervention: ${annual_rev_at_risk:,.2f}')
print('=' * 60)

# 