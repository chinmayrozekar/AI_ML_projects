# EasyVisa — Advanced Machine Learning Project

## Project Summary
EasyVisa is a real-world dataset from a visa-consultancy use-case. This project builds advanced ML solutions to predict outcomes, extract signals from application features, and provide business recommendations to improve processing and customer targeting.

## Business Context & Objective
- Business context: EasyVisa (a visa consultancy) needs to automate and improve decision-making for visa application processing and identify high-risk or high-opportunity applications.
- Objective: Build predictive models to classify or score applications, surface important features, and provide recommendations to streamline operations and reduce manual effort.

## Data
- Files in this folder: `EasyVisa.csv`, `ChinmayRozekar_MachineLearning_Project3_EasyVisa_Full_Code.ipynb`, rendered HTML version of the notebook, and `sample.txt`.
- Dataset size: ~1.8M bytes CSV (check `EasyVisa.csv`); contains application-level features and target variables used for supervised modeling.

## Approach
1. Data ingestion and exploratory data analysis (EDA): distributions, missing-value handling, and feature understanding.
2. Feature engineering: encoding categorical fields, scaling numeric features, deriving business features where applicable.
3. Modeling: experimented with logistic regression, tree-based ensembles (RandomForest, XGBoost), and model explainability (feature importance / SHAP where applicable).
4. Evaluation: precision, recall, ROC-AUC; choose model depending on business objective (precision for targeted campaigns, recall for risk detection).

## Key Findings & Conclusions
- Important predictors: (as found in the notebooks) features related to applicant profile, document completeness, prior history, and key numerical indicators consistently showed high importance.
- Ensemble models (XGBoost / RandomForest) typically produced the best balance between accuracy and robustness on held-out data.
- Use threshold tuning to control trade-offs between precision and recall depending on whether the business prioritizes fewer false positives or catching all positives.

## Recommendations (business actionable)
- Deploy a scoring model to prioritize high-probability applications for expedited processing.
- Use model predictions to route potentially risky applications to a specialized review team to reduce bottlenecks.
- Retrain and recalibrate the model quarterly or after each major campaign or policy change.
- Capture additional signals (document submission timestamps, manual review notes) to improve model performance.

## Technologies & Libraries
- Environment: Jupyter Notebook / Google Colab
- Core libraries: Python, Pandas, NumPy
- Modeling: Scikit-learn, XGBoost, SHAP (optional)
- Visualization: Matplotlib, Seaborn

## Where the code is
- Notebook: `ChinmayRozekar_MachineLearning_Project3_EasyVisa_Full_Code.ipynb` (also rendered as HTML in this folder)
  - Notebook URL: https://github.com/chinmayrozekar/AI_ML_projects/blob/main/Project_3_Advance_Machine_Learning_EasyVisa/ChinmayRozekar_MachineLearning_Project3_EasyVisa_Full_Code.ipynb
  - Rendered HTML: https://github.com/chinmayrozekar/AI_ML_projects/blob/main/Project_3_Advance_Machine_Learning_EasyVisa/ChinmayRozekar_MachineLearning_Project3_EasyVisa_Full_Code.html
- Data: `EasyVisa.csv` — https://github.com/chinmayrozekar/AI_ML_projects/blob/main/Project_3_Advance_Machine_Learning_EasyVisa/EasyVisa.csv

## Quick code snippet (from notebook)
```python
# example: training an XGBoost classifier
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

data = pd.read_csv('EasyVisa.csv')
X = data.drop('target', axis=1)
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)
print('ROC AUC:', roc_auc_score(y_test, model.predict_proba(X_test)[:,1]))
```

---

*Author: Chinmay Rozekar*