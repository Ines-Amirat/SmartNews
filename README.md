## 🔧 1. Tech Stack

- Python
- scikit-learn
- TF-IDF + LinearSVC
- Jupyter Notebook (exploration + analysis)
- Clean project structure (src/, scripts/, data/, models/)

# 📰 2. SmartNews – NLP News Classification  
A machine learning project that classifies news articles into four categories (world, sports, business, sci_tech) using text preprocessing, TF-IDF vectorization, and a Linear SVM model.

This project includes :
- A complete ML pipeline
- A clean file architecture (src/, data/, models/, scripts/, notebooks/)
- A Jupyter notebook for analysis & visualization
- A reusable inference module to make predictions

---

## 🎯 3.  Project Objective

SmartNews aims to demonstrate how to build an end-to-end **text classification system**:

1. Load & prepare real-world news data  
2. Clean and preprocess raw text  
3. Build an ML pipeline (TF-IDF + Linear SVC)  
4. Train and evaluate the model  
5. Save the trained model for later use  
6. Predict the category of new unseen articles  

This project follows a **clean and modular architecture** suitable for academic, research, or internship portfolios.

---

## 🧠 4.  Model & Methodology

### 🔹 Text Preprocessing
- Lowercasing  
- Removal of punctuation  
- Normalization of whitespaces  

### 🔹 Feature Extraction
**TF-IDF Vectorizer** with bi-grams  
- max features = 20,000  
- (n-gram range 1–2)

### 🔹 Classifier
**Linear SVC (Support Vector Classifier)**  
Chosen for:
- Strong performance in text classification  
- Fast training  
- Good generalization on small datasets  

---


---

## 🚀 5. Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone <repository-url>
cd SmartNews


python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

```

This will:

✔ Load the dataset
✔ Preprocess text
✔ Train the TF-IDF + SVM pipeline
✔ Print a classification report
✔ Save the trained model inside models/model.joblib

## 📊 6. Results (Example Classification Report)


              precision    recall  f1-score   support

    business       1.00      1.00      1.00         1
    sci_tech       1.00      1.00      1.00         1
       sports       0.50      1.00      0.67         1
        world       0.00      0.00      0.00         1

    accuracy                           0.75         4
   macro avg       0.62      0.75      0.67         4
weighted avg       0.62      0.75      0.67         4


## 7. Make Predictions

Use the inference module:

``` bash

from src.ml.inference import predict

predict("NASA announces new mission to explore exoplanets.")

```

Example output:

"sci_tech"
