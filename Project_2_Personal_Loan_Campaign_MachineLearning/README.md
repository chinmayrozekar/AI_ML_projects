# Personal Loan Campaign — Project README

## Project Summary
AllLife Bank ran a successful liability-customer campaign and wants to scale marketing for personal loans. This project builds a predictive model to identify customers most likely to accept a personal loan, enabling more efficient targeting and higher conversion with lower cost-per-acquisition.

## Business Context & Objective
- Business context: AllLife Bank wants to increase personal loan adoption among existing liability customers while optimizing marketing spend.
- Objective: Predict which customers will accept the loan offer and identify top features driving conversions to guide targeted campaigns.

## Data
- Source: `Loan_Modelling.csv` (included in this project folder)
- Rows: ~5000 | Key columns: `ID`, `Age`, `Experience`, `Income`, `ZIP Code`, `Family`, `CCAvg`, `Education`, `Mortgage`, `Personal_Loan`, `Securities_Account`, `CD_Account`, `Online`, `CreditCard`

## Approach
1. Data ingestion and EDA: sanity checks, distributions, correlations.
2. Feature engineering and preprocessing: handle categorical encoding, scaling, and splitting into train/test.
3. Modeling: built supervised models (Logistic Regression, Decision Tree, ensemble / evaluation metrics).
4. Model selection and business recommendations: choose a model balancing precision and recall for campaign targeting.

## Key Findings & Conclusions
- Baseline conversion observed in historical campaign was ~9% (useful to benchmark model lift).
- High-impact features (consistently found across models): Income, CCAvg (credit card spend), Education level, Family size, and Existing accounts (Securities/CD).
- A precision-focused model will reduce wasted marketing spend by targeting customers with higher likelihood to convert.

## Recommendations (business actionable)
- Use the model to create a high-probability target list for the next campaign (use precision threshold to control volume).
- A/B test offer creative and incentive levels on a held-out validation cohort to measure incremental lift.
- Monitor cohort performance and retrain model quarterly with new campaign data.
- Consider enrichment (additional third-party data) for customers near the decision boundary to improve accuracy.

## Technologies & Libraries
- Environment: Jupyter Notebook / Google Colab
- Core libraries: Python, Pandas, NumPy
- Modeling & evaluation: Scikit-learn, Matplotlib, Seaborn

## Where the code is
- Notebook: `Personal Loan Campaign.ipynb`
  - Raw notebook: https://github.com/chinmayrozekar/AI_ML_projects/blob/main/Project_2_Personal_Loan_Campaign_MachineLearning/Personal%20Loan%20Campaign.ipynb
- Data: `Loan_Modelling.csv` — https://github.com/chinmayrozekar/AI_ML_projects/blob/main/Project_2_Personal_Loan_Campaign_MachineLearning/Loan_Modelling.csv

## Quick code snippet (from notebook)
```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score

X = data.drop('Personal_Loan', axis=1)
y = data['Personal_Loan']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Precision:', precision_score(y_test, y_pred))
print('Recall:', recall_score(y_test, y_pred))
print('Accuracy:', accuracy_score(y_test, y_pred))
```

---

*Author: Chinmay Rozekar*