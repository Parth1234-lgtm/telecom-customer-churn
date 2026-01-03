# Telecom Customer Churn Prediction

## Project Overview
Customer churn is a critical challenge for telecom companies, as retaining existing customers is often more cost-effective than acquiring new ones. 

This project analyzes customer behavior data and builds a machine learning model to predict whether a customer is likely to churn.The project follows a complete data science workflow, including exploratory data analysis (EDA), data preprocessing, model training, evaluation, and optimization.

##  Dataset
- Source: Kaggle – Telco Customer Churn Dataset
- Size: appprox 7,000 customer records
- Target Variable: Churn (Yes / No)
Each row represents a customer with demographic information, service usage, contract details, and billing data.

## Exploratory Data Analysis (EDA)
Key insights from data exploration include:

- Customers on month-to-month contracts have significantly higher churn rates
- Customers with shorter tenure are more likely to churn
- Higher monthly charges are associated with increased churn risk
- The dataset is class-imbalanced, with fewer churned customers than retained ones

These insights helped guide preprocessing decisions and model design.

## Modeling Approach
- Model Used: Logistic Regression
- Logistic regression was chosen as a strong and interpretable baseline for binary classification.
- Categorical variables were encoded using label encoding and one-hot encoding.
- Numerical features were scaled using Min-Max Scaling.
- Data was split into training and testing sets to ensure unbiased evaluation.

## Handling Class Imbalance
The baseline model showed low recall for churned customers, largely due to class imbalance.

To solve this:

- A class-weighted logistic regression model was trained
- This improved recall for the churn class by penalizing misclassification of minority samples more heavily

## Model Optimization
Two additional steps were applied to improve model performance and interpretability:

- ### Feature Importance
    - Logistic regression coefficients were analyzed to identify the most influential features
    - Contract type, tenure, and monthly charges were found to be the strongest drivers of churn

- ### Threshold Tuning
    - Instead of using the default 0.5 decision threshold, a lower threshold was tested
    - This improved recall for churn prediction at the cost of increased false positives
    - The trade off highlights how business objectives can influence model decisions

## Results Summary
    - Baseline model achieved reasonable accuracy
    - Recall for churned customers improved after class balancing and threshold tuning
    - The final model provides a strong, interpretable baseline for churn prediction

## Technologies Used
    - Python
    - Pandas, NumPy
    - Matplotlib, Seaborn
    - Scikit-learn

## Repository Structure
telecom-customer-churn/

│

├── churn_analysis.ipynb

├── README.md

├── requirements.txt

└── data/
    └── telecom_churn.csv

## Future Improvements
Potential extensions to this project include:

- Trying tree-based models (e.g., Random Forest, Gradient Boosting)
- Applying resampling techniques such as SMOTE,Ensemble etc.
- Performing hyperparameter tuning
- Incorporating customer usage behavior for richer features