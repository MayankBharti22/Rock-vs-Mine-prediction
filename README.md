# Rock vs Mine Prediction Using Machine Learning

This project focuses on classifying sonar signals as **Rock (R)** or **Mine (M)** using multiple machine learning algorithms.  
It demonstrates proper data preprocessing, model comparison, and evaluation on a classic binary classification problem.

---

## 📌 Project Overview

Sonar-based object detection is widely used in underwater navigation and defense systems.  
The goal of this project is to build reliable machine learning models that can distinguish between rocks and underwater mines using sonar signal data.

Key objectives:
- Apply multiple supervised learning algorithms
- Prevent data leakage using pipelines
- Compare models using fair evaluation metrics
- Identify the most suitable classifier for this dataset

---

## 📂 Dataset

- **Name:** Sonar Dataset  
- **Samples:** 208  
- **Features:** 60 numerical attributes  
- **Target Classes:**  
  - `R` → Rock  
  - `M` → Mine  

Each feature represents the energy of a sonar signal at a specific frequency band.

---

## ⚙️ Technologies Used

- Python  
- NumPy  
- Pandas  
- Scikit-learn  
- Jupyter Notebook  

---

## 🧠 Machine Learning Models Implemented

The following models were trained and evaluated:

- Logistic Regression  
- K-Nearest Neighbors (KNN)  
- Support Vector Machine (SVM)  
- Naive Bayes  
- Decision Tree  
- Random Forest  

All applicable models were implemented using **pipelines** to ensure correct preprocessing and to avoid data leakage.

---

## 🔄 Data Preprocessing

- Feature–target separation  
- Train-test split (80:20) with stratification  
- Feature scaling using `StandardScaler`  
- Pipeline-based preprocessing to prevent data leakage  

---

## 📊 Model Evaluation

Models were evaluated using:
- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  

Cross-validation was used where appropriate to ensure stable and reliable results.

---

## 🏆 Results Summary

- **Best Model:** Support Vector Machine (SVM)  
- **Reason:** Performs well on small, high-dimensional datasets with clear class margins  
- Logistic Regression provided a strong and interpretable baseline  
- KNN showed high accuracy but was sensitive to parameter tuning  
- Tree-based models tended to overfit  
- Naive Bayes underperformed due to violated independence assumptions  

---

## 📌 Conclusion

This project highlights the importance of:
- Proper preprocessing
- Avoiding data leakage
- Choosing models based on data characteristics
- Using robust evaluation metrics

Support Vector Machine emerged as the most reliable model for sonar signal classification.

---

## 🚀 Future Improvements

- Dimensionality reduction using PCA  
- Neural network-based models  
- ROC–AUC and Precision–Recall analysis  
- Real-time deployment as a prediction system  

---

## 👤 Author

**Mayank Bharti**

