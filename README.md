# Machine Learning Real-World Projects

This repository contains end-to-end machine learning implementations and business recommendations across three core paradigms: **Supervised Classification**, **Supervised Regression**, and **Unsupervised Clustering**.

---

## Repository Structure
* `Name_ML_Projects.ipynb`: Complete Jupyter Notebook containing data inspection, preprocessing pipelines, model training, validation experiments, test evaluations, and visualizations.
* `TeamName_ML_Report.docx`: Executive business report summarizing methodology, comparative experiment results, visual evidence, recommendations, and limitations.

---

## Projects Overview

### 1. Product Sample Classification (Supervised Learning)
* **Objective:** Classify wine samples into three categories based on 13 chemical measurements to automate laboratory quality sorting.
* **Models Compared:** Majority-class Baseline, K-Nearest Neighbors (`n_neighbors`: 3, 5, 11), and Support Vector Machine (`kernel="rbf"`, `C`: 0.1, 1, 10) with `StandardScaler` fitted strictly on training data.
* **Selected Model:** **SVM (`C=1`, `kernel="rbf"`)**, selected based on the highest validation Macro F1-score.

### 2. Regional Housing Value Estimation (Regression)
* **Objective:** Estimate median residential housing values (`MedHouseVal`, in $100,000s) across California census block groups.
* **Models Compared:** Mean Baseline, Ridge Regression (`alpha`: 0.1, 1, 10), and Random Forest Regressor (`max_depth`: 5, 10, None).
* **Selected Model:** **Random Forest Regressor (`max_depth=None`)**, achieving the lowest validation RMSE and capturing non-linear geographic and economic patterns.

### 3. Wholesale Customer Segmentation (Unsupervised Learning)
* **Objective:** Discover distinct customer purchasing segments across six annual product spending categories (`Fresh`, `Milk`, `Grocery`, `Frozen`, `Detergents_Paper`, `Delicassen`) to design targeted marketing offers.
* **Models Compared:** K-Means and Agglomerative Clustering (`linkage="ward"`) across `k = 2, 3, 4, 5` after `log1p` transformation and `StandardScaler`.
* **Selected Model:** **K-Means (`k=2`)**, achieving the highest Silhouette Score (`0.2903`) and separating customers into *Foodservice/Horeca Buyers* (Cluster 0) and *Retail/Supermarket Bulk Buyers* (Cluster 1).

---

## Tools & Libraries
* **Language:** Python 3
* **Libraries:** `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`
