# Breast Cancer Tumor Detection using Modified AlexNet

## Overview
This project explores **tumor detection in mammogram images** using a **Modified AlexNet Deep Convolutional Neural Network (DCNN)**. The study focuses on improving **classification accuracy** between benign and malignant tumors through **deep learning and data augmentation techniques**.

## Motivation
- **Early Detection**: Breast cancer is one of the leading causes of mortality in women. Early detection significantly increases survival rates.
- **Challenges in Diagnosis**: Traditional machine learning models often struggle with noisy, low-contrast mammogram images.
- **Advancements in AI**: Modified **AlexNet DCNN** improves diagnostic accuracy, aiding radiologists in decision-making.

## Research Objectives
- Enhance accuracy in **tumor classification** in mammograms.
- Implement **data augmentation** to increase dataset size and diversity.
- Modify the **AlexNet architecture** for **binary classification (benign vs. malignant tumors).**
- Optimize the CNN model using **pre-processing and training techniques** to handle noise and resolution variations.

## Methodology

### 1. **Dataset**: MIAS (Mammographic Image Analysis Society)
- Dataset Preprocessing:
  - Applied **Gaussian filtering** to remove noise.
  - Downscaled images from **1024×1024 to 64×64 pixels** for computational efficiency.

### 2. **Data Augmentation**
- **Flipping & Rotations** (90°, 180°, 270°).
- Increased dataset size from **322 to 2,576** images.

### 3. **Modified AlexNet Architecture**
- **Removed the last three layers** of the standard AlexNet.
- Added **Fully Connected Layer + Softmax Layer + Classification Layer**.
- Adjusted **convolutional layers** to suit mammogram imaging.

### 4. **Training & Optimization**
- **Optimizer**: Stochastic Gradient Descent with Momentum (SGDM).
- **Dataset Split**: 31% for Training, 69% for Testing.
- **Evaluation Metrics**:
  - Accuracy: **95.70%**
  - Precision: **0.95**
  - Recall: **0.946**
  - F1 Score: **0.947**
  - ROC Curve AUC: **0.957**

## Critical Analysis
### Strengths:
✔ **High Accuracy** using Modified AlexNet.  
✔ **Improved Generalization** with Data Augmentation.  
✔ **Efficient Training** via optimized preprocessing techniques.  

### Areas for Improvement:
⚠ **Limited Dataset Diversity**: Testing on a larger variety of mammogram images is necessary.  
⚠ **More Evaluation Metrics**: Future studies should analyze additional performance measures (e.g., Sensitivity, Specificity).  

## How to Run the Model

### 1. Clone the Repository
```bash
git clone <repository_url>
cd breast_cancer_tumor_detection
```

### 2. Install Dependencies
```bash
pip install tensorflow numpy pandas matplotlib scikit-learn opencv-python
```

### 3. Run the Model
```bash
python train_model.py
```

### 4. Evaluate Performance
```bash
python evaluate_model.py
```

## Author
**Dhiraj Bandi**  
M.S. Data Science & Artificial Intelligence  
University of Central Missouri  
Email: dhirajbandiwork@gmail.com  

---
This project aims to **leverage AI for early breast cancer detection**, improving diagnostic accuracy and supporting medical professionals.
