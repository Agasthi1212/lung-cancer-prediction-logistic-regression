# ♾️ Lung Cancer Prediction with Logistic Regression


![python](https://img.shields.io/badge/python-3.11.9-red) ![pandas](https://img.shields.io/badge/pandas-2.2.3-orange) ![matplotlib](https://img.shields.io/badge/matplotlib-3.10.0-yellow) ![seaborn](https://img.shields.io/badge/seaborn-0.13.2-green) ![statsmodels](https://img.shields.io/badge/statsmodels-0.14.4-blue) ![scikit_learn](https://img.shields.io/badge/scikit_learn-1.6.1-indigo) 

This project predicts lung cases of lung cancer using logistic regression, applying statistical feature selection and robust evaluation metrics. It demonstrates how clinical and lifestyle factors can be used to classify lung cancer presence, supporting medical decision-making and benefit prioritization. All code and results are in the analysis notebook.

The repository contains the following:

|Description|File|
|:-|:-|
|Analysis notebook with all code used to generate the results, including comments and notes. |  `Lung_Cancer_Prediction_Logistic_Regression.ipynb`|
|Dataset file containing all data used in the analysis. [More Accurate Lung Cancer Dataset (Kaggle)](https://www.kaggle.com/datasets/chandanmsr/more-accurate-lung-cancer-dataset)  | `lcs.csv`|
|Analysis report discussing findings in more detail. | `Lung_Cancer_Prediction_Analysis_Report.pdf`|

## Project Overview

This project explores the use of a logistic regression classification model to predict cases of lung cancer based on a person’s demographics, lifestyle, and symptoms.

- **Dataset:** [More Accurate Lung Cancer Dataset (Kaggle)](https://www.kaggle.com/datasets/chandanmsr/more-accurate-lung-cancer-dataset)  
  911 deduplicated records with binary and numeric features: age, gender, smoking status, symptoms, and lung cancer diagnosis.

- **Objectives:**
  - **EDA:** Assess feature distributions, class balance, and data quality.
  - **Feature Selection:** Use backward elimination (p-values) to retain only statistically significant predictors.
  - **Model Training:** Build and evaluate a logistic regression classifier for binary lung cancer prediction.
  - **Interpretation:** Analyze model performance and provide actionable insights for healthcare applications.

## Key Features

- **Data Cleaning & Preprocessing:**  
  - Removal of duplicates  
  - Binary encoding of categorical variables for model compatibility

- **Exploratory Data Analysis (EDA):**  
  - Visualizations of age and symptom distributions  
  - Class balance checks for lung cancer diagnosis

- **Feature Selection:**  
  - Backward elimination using statsmodels logit regression  
  - Selection of nine key predictors based on statistical significance

- **Model Training & Evaluation:**  
  - Logistic regression with train-test split (70/30)  
  - Metrics: accuracy, precision, recall, F1-score, ROC AUC  
  - Diagnostic plots: ROC curve, confusion matrix

- **Result Interpretation:**  
  - Analysis of feature importance and model reliability  
  - Recommendations for future improvements and subgroup modeling

## Results

- **Key Predictors:**  
  - Smoking status, chest pain, shortness of breath, swallowing difficulty, and related symptoms are the strongest predictors.
  - Gender and age were not statistically significant in this dataset.

- **Model Performance:**  
  - Accuracy: 93.4%  
  - Precision: 97.7%  
  - Recall: 89.5%  
  - F1-score: 93.4%  
  - ROC AUC: 98.5%  
  - Minimal false positives and strong sensitivity for cancer detection

- **Insights:**  
  - The model reliably identifies lung cancer risk based on clinical symptoms and lifestyle factors.
  - Feature selection improves interpretability and reduces overfitting.

## Technologies Used

- Python (Pandas, NumPy, scikit-learn, statsmodels, Matplotlib, Seaborn)
- Jupyter Notebook

## Insights & Impact

- **Actionable Insights:**  
  - Supports early identification of high-risk patients for lung cancer.
  - Provides a transparent framework for benefit prioritization in medical aid schemes.

- **Methodological Value:**  
  - Demonstrates the effectiveness of statistical feature selection in medical classification tasks.
  - Highlights the importance of balanced datasets and robust evaluation metrics.

**Author:** Jishen Harilal  
**LinkedIn:** www.linkedin.com/in/jishen-harilal  
**Contact:** jishen2108@gmail.com  

---

*For more details, see the full analysis in Lung_Cancer_Prediction_Logistic_Regression.ipynb.*
