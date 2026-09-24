# 📊 Telco Customer Churn Analytics and Predictive Modeling Using AI

[![IBM SkillsBuild](https://img.shields.io/badge/IBM_SkillsBuild-Internship-052F5F?style=for-the-badge&logo=ibm)](https://skillsbuild.org/)
[![BharatCares](https://img.shields.io/badge/BharatCares-CSRBOX-orange?style=for-the-badge)](https://bharatcares.org/)
[![AICTE](https://img.shields.io/badge/AICTE-Internship_Portal-blue?style=for-the-badge)](https://internship.aicte-india.org/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/scikit_learn-Machine_Learning-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)]()

---

## 📌 Project Overview
Customer attrition (churn) is a major revenue drain in the subscription-based telecommunications industry. Acquiring a replacement subscriber costs **5 to 7 times more** than retaining an existing customer.

This repository contains the complete end-to-end Data Analytics and Artificial Intelligence project developed as part of the **IBM SkillsBuild Data Analytics with AI Academic Internship Program**, conducted by **BharatCares** in association with **AICTE**.

The objective of this project is to build an interpretable, high-performance predictive engine capable of:
1. Detecting early behavioral and contractual signals of customer churn.
2. Quantifying attrition probabilities for every customer account.
3. Enabling proactive, AI-driven retention strategies to preserve recurring subscription revenue.

---

## 🔗 Dataset Information & Link
- **Dataset Name:** IBM Telco Customer Churn Dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`)
- **Direct Dataset Link:** [IBM Telco Customer Churn on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Official IBM Repository:** [IBM Developer Telco Churn Repository](https://github.com/IBM/telco-customer-churn-eda-and-model-inline)
- **Instances:** 7,043 customer accounts
- **Attributes:** 21 multi-dimensional features spanning Demographics, Account & Billing, and Subscribed Services
- **Target Variable:** `Churn` (`Yes` / `No`) — Baseline attrition rate is **26.54%** (1,869 churned vs. 5,174 retained)

---

## 🏆 Key Machine Learning & AI Results
Three complementary supervised machine learning architectures were trained on an 80/20 stratified train-test split (1,409 unseen test customers):

| Machine Learning Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **80.62%** | 65.93% | **55.88%** | **60.49%** | 0.8422 |
| **Random Forest Classifier** | 79.99% | **66.55%** | 49.47% | 56.75% | **0.8441** |
| **Gradient Boosting Classifier** | 79.91% | 64.82% | 53.21% | 58.44% | 0.8425 |

- **Top Model Discrimination:** **Random Forest** and **Gradient Boosting** achieved an outstanding **0.844 ROC-AUC**.
- **Balanced Diagnostic:** **Logistic Regression** achieved the highest overall accuracy (**80.62%**) and highest Recall (**55.88%**).

---

## 🔍 Key Data Insights & Discoveries
1. **Contract Duration:** Month-to-month contracts exhibit a severe **42.71% churn rate**, whereas 1-year contracts drop to **11.27%** and 2-year contracts achieve an outstanding **2.83%** retention rate.
2. **Tenure Vulnerability:** The first **12 months** represent the highest attrition risk; churn probability decays rapidly as customer tenure matures.
3. **Payment Methods:** Subscribers utilizing **Electronic Checks** churn at **45.29%**, compared to only **~15%–16%** for automated bank transfers and credit cards.
4. **The Fiber Optic Paradox:** Fiber optic subscribers experience an elevated churn rate (**41.89%**), driven by higher median monthly charges and service reliability expectations.
5. **Protective Add-ons:** Value-add services like **Tech Support** and **Online Security** slash churn rates by more than **25 percentage points**.

---

## 💡 Prescriptive Business Recommendations
- **Incentivize 1-Year/2-Year Contract Migrations:** Offer a 10% monthly rebate or complimentary device protection for upgrading from month-to-month terms.
- **Early-Tenure Onboarding Protocol:** Establish proactive customer success check-ins at Days 14, 45, and 90 to guide new subscribers.
- **Bundle Online Security & Tech Support:** Package these protective services into entry-level fiber optic plans to mitigate the 41.9% churn rate.
- **Digital Auto-Pay Adoption:** Provide a recurring $5 monthly bill credit for enabling automated bank or credit card payments.
- **Financial ROI Impact:** Intervening on high-confidence churn customers (predicted probability $\ge$ 70%) protects **over $143,000 in annual recurring revenue** per 1,000 at-risk subscribers.

---

## 🛠️ Technologies Used
- **Programming Language:** Python 3.10+
- **Data Manipulation & Analysis:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Machine Learning & Modeling:** Scikit-Learn (LogisticRegression, RandomForestClassifier, GradientBoostingClassifier, StandardScaler)
- **Documentation & Reporting:** Jupyter Notebook, python-docx, Markdown

---
```