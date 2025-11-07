# Bank Churn Prediction — Project README

## Project Summary
A neural-network–based classifier to predict whether a bank customer will leave (churn) within six months. The project focuses on identifying drivers of churn so the bank can take proactive retention actions.

## Business Context & Objective
- Business context: Customer churn reduces recurring revenue and increases acquisition costs. Banks need early-warning systems to retain valuable customers.
- Objective: Build and evaluate models that predict churn; surface the most important features and produce business actions to reduce churn.

## Data
- Key files in this folder: `Project4_BankChurnPrediction_IntroductionToNeuralNetworks_ChinmayRozekar.ipynb` (notebook), rendered HTML, and `bank-1.csv` (dataset).
- Typical columns: `CustomerId`, `Surname`, `CreditScore`, `Geography`, `Gender`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary`, `Exited` (target).

## Approach
1. Data loading and sanity checks (shape, dtypes, missing values).  
2. Exploratory analysis to identify distributions and correlations.  
3. Preprocessing: encoding categorical variables, scaling numeric features, dealing with class imbalance (SMOTE).  
4. Modeling: Neural network (Keras/TensorFlow) classifier with tuning; alternative baselines (tree-based models) for comparison.  
5. Evaluation: precision, recall, F1, ROC-AUC and confusion matrices; choose operating threshold based on business cost trade-offs.

## Key Findings & Conclusions
- Strong predictors: Tenure, Balance, NumOfProducts, Age, CreditScore, and IsActiveMember.  
- Addressing class imbalance (SMOTE) improved recall while preserving acceptable precision.  
- Neural networks produced competitive ROC-AUC and can be tuned for a precision-conscious retention campaign or a recall-focused risk-detection workflow.

## Recommendations (business actionable)
- Flag high-risk customers for retention offers (personalized outreach, incentives).  
- Route flagged customers to a retention team with tailored offers and monitor uplift via A/B tests.  
- Retrain or recalibrate models regularly (monthly/quarterly) to incorporate new customer behaviour and campaign outcomes.  
- Use model explainability (feature importance / SHAP) to guide product and loyalty program changes.

## Technologies & Libraries
- Python, Pandas, NumPy, Matplotlib, Seaborn  
- TensorFlow / Keras, scikit-learn, imbalanced-learn (SMOTE)

## Where the code is
- Notebook: `Project4_BankChurnPrediction_IntroductionToNeuralNetworks_ChinmayRozekar.ipynb`  
  - HTML render: https://github.com/chinmayrozekar/AI_ML_projects/blob/main/Project_4_Bank_Churn_Prediction/Project4_BankChurnPrediction_IntroductionToNeuralNetworks_ChinmayRozekar.html  
  - Notebook: https://github.com/chinmayrozekar/AI_ML_projects/blob/main/Project_4_Bank_Churn_Prediction/Project4_BankChurnPrediction_IntroductionToNeuralNetworks_ChinmayRozekar.ipynb  
- Data: `bank-1.csv` — https://github.com/chinmayrozekar/AI_ML_projects/blob/main/Project_4_Bank_Churn_Prediction/bank-1.csv

## Quick code snippet (from notebook)
```python
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# preprocessing
X = ds.drop('Exited', axis=1)
y = ds['Exited']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# handle imbalance
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)

# simple NN
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_res.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
model.fit(X_res, y_res, epochs=20, batch_size=128, validation_split=0.1)
```