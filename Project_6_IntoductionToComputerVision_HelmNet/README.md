# HelmNet — Computer Vision Project README

## Project Summary
HelmNet is an image-classification pipeline to detect whether workers are wearing safety helmets (with-helmet vs without-helmet). The model supports workplace safety automation in industrial environments.

## Business Context & Objective
- Business context: Automating PPE (personal protective equipment) compliance reduces accidents and safety violations.  
- Objective: Build a robust image classifier, evaluate performance under variable lighting/angles, and provide deployment-ready artifacts for inference.

## Data
- Dataset: 631 RGB images (balanced across classes), stored as NumPy arrays in the notebook (`images_proj.npy`) and labels in `Labels_proj.csv`.  
- Files: `Project_6_ChinmayRozekar_HelmNet_Full_Code.ipynb`, rendered HTML, and image/data files. Example shapes: (631, 200, 200, 3).

## Approach
1. Data loading and visualization; verify image shapes and label balance.  
2. Data augmentation via ImageDataGenerator (rotation, flips, brightness) to improve generalization.  
3. Model: CNN built with Keras (Conv2D, MaxPooling, BatchNorm, Dropout) and optionally transfer learning (VGG16).  
4. Evaluation: accuracy, F1-score, confusion matrix; visualize misclassified examples for error analysis.

## Key Findings & Conclusions
- Data augmentation improved generalization on small dataset sizes.  
- Transfer learning (VGG16-based) accelerated convergence and improved final performance compared to training from scratch.  
- Model performs well in controlled environments; further domain adaptation needed for deployment in varied real-world sites.

## Recommendations (business actionable)
- Integrate real-time inference pipeline at edge (optimized model using TensorFlow Lite or ONNX for embedded devices).  
- Collect more labeled images from target sites (different lighting and camera angles) and retrain for improved robustness.  
- Add alerting/logging pipeline for non-compliance events and human review of ambiguous cases.

## Technologies & Libraries
- Python, NumPy, Pandas, Matplotlib, OpenCV  
- TensorFlow / Keras, scikit-learn

## Where the code is
- Notebook: `Project_6_ChinmayRozekar_HelmNet_Full_Code.ipynb`  
  - HTML render: https://github.com/chinmayrozekar/AI_ML_projects/blob/main/Project_6_IntoductionToComputerVision_HelmNet/Project_6_ChinmayRozekar_HelmNet_Full_Code.html  
  - Notebook: https://github.com/chinmayrozekar/AI_ML_projects/blob/main/Project_6_IntoductionToComputerVision_HelmNet/Project_6_ChinmayRozekar_HelmNet_Full_Code.ipynb

## Quick code snippet (from notebook)
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(200,200,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

---

*Author: Chinmay Rozekar*
