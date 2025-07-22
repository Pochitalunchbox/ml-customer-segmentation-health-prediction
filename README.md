# Machine Learning for Customer Segmentation, Housing Price Prediction, and Disease Classification

This repository demonstrates the use of machine learning to address three real-world problems:

- 🧩 Customer Segmentation using clustering techniques
- 🏡 Housing Price Prediction using regression models
- 🩺 Liver Disease Detection using classification algorithms

It includes clean, modular code with insightful visualizations, making it suitable for both technical showcase and practical application.

---

## 📁 Project Structure

- **Clustering (Mall Customers)**
  - K-Means Clustering
  - Gaussian Mixture Models (GMM)
  - Hierarchical Agglomerative Clustering (HAC)

- **Regression (California Housing Prices)**
  - Linear Regression
  - Polynomial Regression
  - Lasso Regression
  - Random Forest Regressor

- **Classification (Liver Disease Prediction)**
  - Logistic Regression
  - Support Vector Classifier (SVC)
  - K-Nearest Neighbors (KNN)
  - Random Forest Classifier
 
  - 
---

## 🧩 1. Customer Segmentation (Clustering)

**Dataset:** Mall Customers  
**Features Used:** Age, Annual Income (k$), Spending Score (1–100)

### 🔍 Key Insights

- Cluster 0: Young with low income but high spending behavior
- Cluster 1: Middle-aged, high income, high spending
- Cluster 2: Older, high income, low spending

### 📊 Visualizations

- Correlation matrix
- Age, income, and spending score distributions
- Elbow Method for K optimization
- Box plots by cluster (Age, Income, Spending)
- GMM clustering box plots
- HAC dendrogram and clustering box plots

---

## 🏡 2. Housing Price Prediction (Regression)

**Dataset:** California Housing Dataset  
**Target Variable:** Median House Value  
**Goal:** Predict housing prices using socioeconomic and geographic data

### 🔧 Models & Performance

| Model                  | R² Score | RMSE         |
|------------------------|----------|--------------|
| Linear Regression      | 0.6485   | 69,297.72    |
| Polynomial Regression  | 0.7085   | 63,122.37    |
| Lasso Regression       | 0.6488   | 69,297.12    |
| Random Forest Regressor| 0.8261   | 43,767.99    |

### 🔍 Feature Importance (Random Forest)

Top influencing factor: `median_income`  
Other factors: location-based features and total rooms

---

## 🩺 3. Liver Disease Classification (Supervised Learning)

**Dataset:** Liver Patient Records  
**Target Variable:** Presence of Liver Disease (0 = No, 1 = Yes)

### ⚙️ Models Used & Evaluation

| Model                  | ROC AUC | Accuracy | F1 Score |
|------------------------|---------|----------|----------|
| K-Nearest Neighbors    | 1.0000  | 0.9805   | 0.9661   |
| Logistic Regression    | 0.7528  | 0.6335   | 0.5612   |
| Support Vector Machine | 0.7598  | 0.5974   | 0.5494   |
| Random Forest Classifier| 0.9997 | 0.9998   | 0.9997   |

> ⚠️ Note: The high performance of the Random Forest model may be influenced by the structure or imbalance of the dataset. Model generalizability should be evaluated on external data.

### 📊 Visuals

- Distribution of target variable and correlation matrix
- ROC curves and confusion matrices for each model
- Feature importance (Random Forest)

---

## 🧠 Key Learnings

- Effective segmentation using K-Means, GMM, and HAC provides clear customer personas for targeted marketing.
- Ensemble methods like Random Forest significantly outperform linear models in capturing non-linear patterns in housing price data.
- Feature importance visualization and ROC analysis guide model interpretability and trust, especially in health-related classification.

---

## 📎 Tools & Libraries

- Python (Pandas, NumPy, Seaborn, Matplotlib, Scikit-learn)
- Jupyter Notebook
- Data Preprocessing & Feature Engineering
- Model Evaluation: R², RMSE, Accuracy, F1 Score, ROC AUC

---

## 🔗 Connect

[LinkedIn – Ethan Choo](https://www.linkedin.com/in/ethanchoo5/)

