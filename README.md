# Classification-Model-For-Loan-Default-Prediction

Overview
This repository contains the code and analysis for predicting loan defaults using various machine learning models. The project includes data preprocessing, model training, evaluation, and variable importance analysis.


**Introduction**

This project focuses on predicting loan defaults using a variety of machine-learning models. The analysis includes both linear and non-linear models, to identify the most effective approach for robust loan default prediction.


**Exploratory Data Analysis (EDA)**

The initial exploration of the dataset involved several key steps:

Data Import: The dataset was loaded, and a summary was generated to understand its structure and contents.

Missing Values: The presence of missing values in the dataset was checked, and the total count of missing values was reported.

Duplicate Rows: Duplicate rows were identified and examined.

Data Cleaning: The "LoanID" column, assumed to contain unique identifiers, was removed from the dataset as it didn't contribute to predictive modeling.

Variable Separation: The dataset was split into numerical and categorical columns for further analysis.

Visualization: Exploratory visualizations, such as bar plots, were created to understand the distribution of categorical variables.

Dummy Variables: Dummy variables were generated for categorical columns.

Near-Zero Variance: Near-zero variance variables in numerical columns were identified.


**Linear Models**

Logistic Regression:-
The logistic regression model exhibited promising discrimination ability with an ROC of 0.7601. However, its specificity was relatively low at 0.0699, indicating a challenge in correctly identifying non-default instances.

Linear Discriminant Analysis (LDA):-
LDA demonstrated excellent sensitivity at 0.9948, but its specificity was notably low at 0.0554.

Partial Least Squares Discriminant Analysis (PLSDA):-
PLSDA achieved perfect sensitivity but struggled with specificity (0.0000), suggesting potential overfitting.

Penalized Regression:-
Penalized regression, incorporating both Lasso and Ridge regularization (alpha=0.1, lambda=0.01), delivered a high sensitivity of 0.9967, but its specificity was modest at 0.0313.

Nearest Shrunken Centroids:-
Nearest Shrunken Centroids, despite achieving perfect sensitivity, faced challenges in specificity (0.0000). The optimal model, considering a balance between sensitivity and specificity, is the logistic regression model.

**Non-Linear Models**

Regularized Discriminant Analysis (RDA)
RDA exhibited a moderate ROC of 0.5706 with a sensitivity of 0.707, but its specificity faced challenges at 0.3738.

Support Vector Machines (SVM):-
SVM presented a higher ROC of 0.6829, with exceptional sensitivity at 0.9984 but minimal specificity at 0.0024.

k-Nearest Neighbors (KNN):-
KNN demonstrated a ROC of 0.6377, showcasing high sensitivity at 0.9997 and complete specificity.

Neural Networks:-
Neural Networks achieved a promising ROC of 0.7532, with sensitivity at 0.9902 and specificity at 0.0867.

Flexible Discriminant Analysis (FDA):-
FDA provided a ROC of 0.7471, sensitivity of 0.9833, and specificity of 0.0723.

Naive Bayes:-
Naive Bayes provided a ROC of 0.7352, sensitivity of 0.9577965, and specificity of 0.0817641.

**Model Evaluation and Comparison**

The top-performing models from both linear and non-linear categories are Logistic Regression and Penalized Regression. Further evaluation and comparison are necessary to identify the most suitable model for loan default prediction.

**Variable Importance**

Variable importance analysis was conducted for both Logistic Regression and Penalized Regression models. The results highlighted key predictors contributing to the understanding of loan default risk.

**Conclusion**

In conclusion, after comprehensive analysis and evaluation, the Logistic Regression model stands out as the most effective for predicting loan defaults. It exhibits a balanced approach with high sensitivity and reasonable specificity. Further optimization and validation on an independent test set are recommended to ensure the model's generalization and robustness.
