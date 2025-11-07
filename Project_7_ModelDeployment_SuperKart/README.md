# SuperKart — Model Deployment Project README

## Project Summary
End-to-end sales forecasting and model deployment for SuperKart retail. The project builds predictive models for product-store sales and demonstrates deployment artifacts (notebook, API patterns, and front/back-end components) for operationalization.

## Business Context & Objective
- Business context: Accurate sales forecasts allow optimized inventory and procurement, reducing stockouts and overstock.  
- Objective: Train robust forecasting/regression models and create a reproducible deployment pipeline (serialization, API, front-end demo).

## Data
- Files: `Project_7_ChinmayRozekar_SuperKart_Model_Deployment_Notebook.ipynb` (notebook + HTML), `SuperKart.csv`, supporting diagrams, frontend_files/ and backend_files/ directories, and deployment notes (`Troubleshooting.md`).
- Data: product features, store metadata, historical sales (`Product_Store_Sales_Total`) and other attributes.

## Approach
1. EDA and feature engineering: Product, store, and location-level aggregations.  
2. Modeling: ensemble regressors (RandomForest, GradientBoosting, XGBoost) and pipelines for preprocessing.  
3. Model selection and serialization (joblib); create a REST API wrapper (Flask) and a simple frontend to demo predictions.  
4. Deployment notes and troubleshooting included for reproducibility.

## Key Findings & Conclusions
- Ensemble regressors (XGBoost / RandomForest) achieved the best accuracy and stability for sales prediction.  
- Product type, store size, allocated display area, and MRP were important contributors to revenue.  
- Packaging a model behind an API and adding a lightweight frontend allowed non-technical stakeholders to query forecasts.

## Recommendations (business actionable)
- Integrate the serialized model with CI/CD for automated retraining and deployment.  
- Monitor prediction drift and business KPIs (forecast error by product/store) and schedule periodic retraining.  
- Expose the API internally for demand-planning teams and connect outputs to inventory planning workflows.

## Technologies & Libraries
- Python, Pandas, NumPy, scikit-learn, XGBoost, joblib  
- Flask for API; simple frontend (HTML/JS) for demo; Hugging Face Hub artifacts for optional sharing

## Where the code is
- Notebook: `Project_7_ChinmayRozekar_SuperKart_Model_Deployment_Notebook.ipynb`  
  - HTML render: https://github.com/chinmayrozekar/AI_ML_projects/blob/main/Project_7_ModelDeployment_SuperKart/Project_7_ChinmayRozekar_SuperKart_Model_Deployment_Notebook.html  
  - Notebook: https://github.com/chinmayrozekar/AI_ML_projects/blob/main/Project_7_ModelDeployment_SuperKart/Project_7_ChinmayRozekar_SuperKart_Model_Deployment_Notebook.ipynb  
- Data: `SuperKart.csv` — https://github.com/chinmayrozekar/AI_ML_projects/blob/main/Project_7_ModelDeployment_SuperKart/SuperKart.csv

## Quick code snippet (from notebook)
```python
# example: load model and expose a flask prediction endpoint
import joblib
from flask import Flask, request, jsonify

model = joblib.load('model.joblib')
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    payload = request.json  # expect features as JSON
    X = pd.DataFrame([payload])
    pred = model.predict(X)
    return jsonify({'prediction': pred[0]})
```

---

*Author: Chinmay Rozekar*
