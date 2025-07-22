# Customer Segmentation, Housing Price Regression, and Liver Disease Prediction

This repository showcases a comprehensive machine learning project involving **Unsupervised Learning (Clustering)**, **Regression Analysis**, and **Classification Models**. The project emphasizes effective data preprocessing, model interpretability, and business applicability across marketing, real estate, and healthcare domains.

## 🔍 1. Customer Segmentation (Unsupervised Learning)

We used clustering techniques to identify high-value customer segments using demographics and behavioral features.

### 1.1 Exploratory Data Analysis (EDA)
![Customer EDA](results/customer_eda.png.PNG)

### 1.2 K-Means Clustering
- Optimal number of clusters determined using Elbow Method.
- Box plots show segment-specific differences in age, income, and spending score.

![KMeans](results/kmeans_cluster_analysis.png.PNG)

### 1.3 Gaussian Mixture Model (GMM)
- Captures soft clustering probabilities.
- Reveals more nuanced patterns in customer segments.

![GMM](results/gmm_cluster_analysis.png.PNG)

### 1.4 Hierarchical Agglomerative Clustering (HAC)
- Dendrogram visually represents cluster linkage hierarchy.

![HAC](results/hac_dendrogram_clusters.png.PNG)

---

## 🏠 2. Housing Price Prediction (Regression)

We used four regression models to predict median housing prices in California based on census features.

### 2.1 Distribution of Median Price
![Housing Distribution](results/housing_price_distribution.png.PNG)

### 2.2 Model Comparison

| Model                    | R² Score | RMSE        |
|-------------------------|----------|-------------|
| Linear Regression       | 0.6485   | 69297.72    |
| Polynomial Regression   | 0.7085   | 63122.37    |
| Lasso Regression        | 0.6488   | 69297.12    |
| Random Forest Regressor | 0.8261   | 48767.94    |

#### Visual Comparison:
- Polynomial regression improves over linear.
- Random Forest shows best fit and lowest error.

![Linear vs Polynomial](results/linear_vs_polynomial.png.PNG)  
![Lasso](results/lasso_regression.png.PNG)  
![Random Forest](results/rf_regression_results.png.PNG)

---

## 🧬 3. Liver Disease Prediction (Classification)

This medical classification task predicts the presence of liver disease using various blood indicators.

### 3.1 Data Insights
- Dataset imbalance is addressed.
- Key liver-related features identified.

![Liver EDA](results/liver_disease_eda.png.PNG)

### 3.2 Model Evaluation

| Model              | ROC AUC | Accuracy | F1 Score |
|-------------------|---------|----------|----------|
| KNN               | 1.0000  | 0.9805   | 0.9661   |
| Logistic Reg.     | 0.7528  | 0.6335   | 0.5612   |
| SVC               | 0.7598  | 0.5974   | 0.5494   |
| Random Forest     | 0.9997  | 0.9998   | 0.9997   |

> ⚠️ *Note: The Random Forest model's near-perfect score likely reflects overfitting due to the nature of the dataset sourced online. A disclaimer is included in the notebook.*

### ROC Curves & Confusion Matrices:

![KNN](results/knn_results.png.PNG)  
![Logistic](results/logistic_results.png.PNG)  
![SVC](results/svc_results.png.PNG)  
![Random Forest Classification](results/rf_classification_results.png.PNG)

### Feature Importance (Random Forest)
![RF Importance](results/rf_classification_importance.png.PNG)

---

## 🔗 Connect

📎 [LinkedIn – Ethan Choo](https://www.linkedin.com/in/ethanchoo5/)


