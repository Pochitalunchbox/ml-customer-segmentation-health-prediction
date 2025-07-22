# 🧠 ML Customer Segmentation & Health Prediction

This repository presents a multi-domain machine learning project that simulates real-world business applications through three core tasks: **customer segmentation**, **housing price prediction**, and **disease classification**. It demonstrates end-to-end data science workflows—from data cleaning to model interpretation—using diverse datasets from retail, real estate, and healthcare sectors.

---

## 💼 Problem Context

This project tackles three business-driven problems:

1. **Retail**: How can businesses segment customers for targeted marketing?
2. **Real Estate**: Can we predict house prices based on neighborhood and demographic features?
3. **Healthcare**: Is it possible to predict liver disease using patient diagnostic data?

By applying clustering, regression, and classification techniques, this project delivers actionable insights for decision-makers across industries.

---

## 📌 Executive Summary

- **Clustering (Unsupervised Learning)**: Identified high-value customer groups using K-Means, GMM, and HAC.
- **Regression (Supervised Learning)**: Built and compared four models to predict California housing prices.
- **Classification (Supervised Learning)**: Developed multiple classifiers to predict liver disease presence.

Models were evaluated using **accuracy, ROC AUC, R², and F1 Score**, with business-oriented insights extracted from each task.

---

## 🛠️ Tech Stack

- **Languages & Libraries**: Python, Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib
- **Machine Learning**: K-Means, GMM, HAC, Linear/Polynomial/Lasso/Random Forest Regressor, Logistic Regression, SVC, KNN, Random Forest Classifier
- **Metrics**: Accuracy, R², F1 Score, ROC AUC, Confusion Matrix

---

## 📁 Project Structure

- `clustering/` — Mall Customer Segmentation
- `regression/` — California Housing Price Prediction
- `classification/` — Liver Disease Classification
- `results/` — Visualizations & performance metrics

---

## 🌟 Key Results Summary

| Task                    | Best Model               | Key Metric     | Score     |
|-------------------------|--------------------------|----------------|-----------|
| Customer Segmentation  | K-Means                  | Optimal Clusters | 3        |
| Housing Price Prediction | Random Forest Regressor | R² Score       | 0.92      |
| Liver Disease Detection | Random Forest Classifier | ROC AUC        | 0.93      |

---

## 📊 Visual Highlights & Business Insights

### 1. Customer Segmentation (K-Means)
![K-Means](results/kmeans_cluster_analysis.png.PNG)  
> Revealed 3 key segments—high-income, high-spending customers were ideal targets for loyalty campaigns.

### 2. GMM & HAC Clustering
![GMM](results/gmm_cluster_analysis.png.PNG)  ![HAC](results/hac_dendrogram_clusters.png.PNG)  
> GMM revealed overlapping segments suggesting the need for soft, adaptive marketing. HAC validated these clusters.

### 3. Random Forest Housing Regression
![RF Regressor](results/rf_regression_results.png.PNG)  
> Outperformed other models (R² = 0.92). Income and location were the strongest price predictors.

### 4. Lasso Regression Feature Shrinkage
![Lasso](results/lasso_regression.png.PNG)  
> Reduced overfitting and isolated high-impact features like median income.

### 5. Liver Disease Prediction (Random Forest)
![RF Classifier](results/rf_classification_results.png.PNG)  
> Achieved ROC AUC of 0.93. Medical markers like Alkaline Phosphatase were top predictors.

---

## ⚠️ Limitations & Future Work

- **Imbalanced classes** in the liver dataset affected model fairness. Future versions will apply SMOTE or cost-sensitive learning.
- The models are validated on held-out test sets but would benefit from **real-world A/B testing**.
- Adding a **Streamlit app** or **dashboard** for stakeholder-facing insights is in the roadmap.

---

## 🔗 Author

**Ethan Choo**  
📍 Singapore | 🎓 Data Science & Business Analytics Graduate (SIM-UOL)  
🔗 [LinkedIn](https://www.linkedin.com/in/ethanchoo5/) | 🔗 [GitHub](https://github.com/ethan-analytics)

---

> If you found this useful, feel free to ⭐ the repo or connect with me!
