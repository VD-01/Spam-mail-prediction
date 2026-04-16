# Spam Email Classification using Machine Learning

## Overview

This project focuses on building a **Spam Email Detection System** using machine learning techniques. The system classifies emails into **Spam (1)** or **Not Spam (0)** using **TF-IDF vectorization** and the **Multinomial Naive Bayes algorithm**.

It helps in automatically filtering unwanted emails and improving email security.

---

## Dataset

The dataset used for this project is from Kaggle:
https://www.kaggle.com/datasets/bayes2003/emails-for-spam-or-ham-classification-trec-2007?resource=download

* Contains labeled email messages
* Classes:

  * **Spam (1)**
  * **Not Spam (0)**
* Real-world dataset (TREC 2007)

---

## Technologies Used

* Python
* Pandas
* Scikit-learn
* NumPy
* Pickle

---

## Working of the Project

### 1. Data Preprocessing

* Remove missing values
* Convert text to lowercase
* Clean and normalize text

### 2. Feature Extraction

* Convert text into numerical vectors using **TF-IDF**
* Limit features to 10,000 for efficiency

### 3. Model Training

* Algorithm: **Multinomial Naive Bayes**
* Train-test split: **80:20**

### 4. Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

---

## Results

* **Accuracy:** 97.97%

### Classification Report:

| Class        | Precision | Recall | F1-score |
| ------------ | --------- | ------ | -------- |
| Not Spam (0) | 0.96      | 0.99   | 0.97     |
| Spam (1)     | 0.99      | 0.97   | 0.98     |

### Confusion Matrix:

```
[[ 9673   120 ]
 [  403 15622 ]]

## How to Run

### 1. Train the Model

```
python train.py
```

### 2. Run Prediction

```
python predict.py "Your message here"
```

If no input is provided, default test messages will be used.

---

## 🔹 Example Output

```
Prediction: [Spam] (Confidence: 98.3%)
```

