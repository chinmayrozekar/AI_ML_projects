# Introduction to Neural Networks — Additional Side Projects

This folder contains compact, hands-on notebooks that introduce neural networks and core ML workflows through practical case studies and exercises. Each notebook is a standalone mini-project suitable for a portfolio entry: business context, data exploration, model building, and concise conclusions.

Contents
- Audio MNIST digit recognition — audio classification with neural nets  
  Notebook: https://github.com/chinmayrozekar/AI_ML_projects/blob/main/Additional%20Side%20Projects/Introduction_to_Neural_Networks/Audio_MNIST_Digit_Recognition.ipynb

- Admission chances prediction — classification/regression case study (student admission)  
  Notebook: https://github.com/chinmayrozekar/AI_ML_projects/blob/main/Additional%20Side%20Projects/Introduction_to_Neural_Networks/Case_Study_Predicting_Chances_of_Admission_updated.ipynb

- Used car price prediction — regression with feature engineering and ensembles / NN  
  Notebook: https://github.com/chinmayrozekar/AI_ML_projects/blob/main/Additional%20Side%20Projects/Introduction_to_Neural_Networks/Case_Study_Used_Car_Price_Prediction.ipynb

- Credit card fraud detection — anomaly / classification workflow, imbalance handling  
  Notebook: https://github.com/chinmayrozekar/AI_ML_projects/blob/main/Additional%20Side%20Projects/Introduction_to_Neural_Networks/Credit_card_Fraud_detection_Notebook_Week.ipynb

- Flight price prediction — time/feature-driven regression and modeling choices  
  Notebook: https://github.com/chinmayrozekar/AI_ML_projects/blob/main/Additional%20Side%20Projects/Introduction_to_Neural_Networks/Flight_Price.ipynb

- INN Low-code learner notebook — low-code introduction to neural nets and exercises  
  Notebook: https://github.com/chinmayrozekar/AI_ML_projects/blob/main/Additional%20Side%20Projects/Introduction_to_Neural_Networks/INN_Learner_Notebook_Low_code.ipynb

- Job change prediction (case study) — classification and feature analysis for HR use-cases  
  Notebook: https://github.com/chinmayrozekar/AI_ML_projects/blob/main/Additional%20Side%20Projects/Introduction_to_Neural_Networks/Session_Notebook_Job_Change_Prediction_Case_Study.ipynb

Quick overview (what each notebook demonstrates)
- Audio_MNIST_Digit_Recognition.ipynb  
  - Task: audio -> spectrogram -> CNN classifier to recognize spoken digits.  
  - Highlights: audio preprocessing (librosa), spectrograms, CNN architecture, data augmentation for audio.

- Case_Study_Predicting_Chances_of_Admission_updated.ipynb  
  - Task: predict admission likelihood from applicant features.  
  - Highlights: EDA, feature transforms, small neural net classifier/regression, model interpretation.

- Case_Study_Used_Car_Price_Prediction.ipynb  
  - Task: regression to predict used-car prices.  
  - Highlights: feature engineering (text / categorical), regression baselines, neural net regression, error analysis.

- Credit_card_Fraud_detection_Notebook_Week.ipynb  
  - Task: detect fraudulent transactions.  
  - Highlights: class imbalance handling (SMOTE/oversampling), evaluation with precision/recall/AUC, thresholding for business trade-offs.

- Flight_Price.ipynb  
  - Task: predict flight ticket prices using time- and route-related features.  
  - Highlights: temporal features, feature encoding, regression modeling and error metrics.

- INN_Learner_Notebook_Low_code.ipynb  
  - Task: low-code walkthrough for learners to experiment with NN concepts.  
  - Highlights: reproducible exercises, minimal code to show core NN building blocks.

- Session_Notebook_Job_Change_Prediction_Case_Study.ipynb  
  - Task: classify job-change likelihood from profile and engagement features.  
  - Highlights: feature importance, class balancing, end-to-end pipeline example.

Tech stack & libraries used
- Core: Python, NumPy, Pandas  
- Modeling & DL: scikit-learn, TensorFlow / Keras (or PyTorch in some experiments), XGBoost (where used)  
- Audio: librosa (spectrograms & preprocessing)  
- Visualization: Matplotlib, Seaborn  
- Notebook environment: Jupyter / Google Colab

How to open the notebooks
1. Open the notebook link above on GitHub to view the rendered output, or clone the repo and run locally.  
2. To run locally: create a Python 3.8+ environment, install the requirements below, then open the notebook with `jupyter notebook`.

Minimal requirements (examples)
```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
# optional for deep learning and audio:
pip install tensorflow librosa xgboost imbalanced-learn
```

Suggested README frontmatter for each notebook when used as a portfolio card
- One-line business problem  
- Tools & libraries (bullet list)  
- Key result (single metric or insight)  
- Link to notebook and a 1–2 sentence takeaway

Representative code snippet (generic training loop)
```python
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
                    Dense(32, activation='relu'),
                    Dense(output_units, activation='sigmoid')])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.1)
```

Notes
- Each notebook is self-contained and includes the code, EDA, and conclusions.  
- Use the notebook links above to review the full code and outputs for each mini-project.  
- For portfolio presentation, consider adding a short thumbnail image and two screenshots (hero plot + code snippet) per notebook to improve recruiter readability.

Author
- Chinmay Rozekar
