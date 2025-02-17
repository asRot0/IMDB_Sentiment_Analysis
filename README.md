# IMDB Sentiment Analysis

## Project Overview
This project performs sentiment analysis on **IMDB movie reviews** using **K-Nearest Neighbors (KNN), Support Vector Machine (SVM), Random Forest, and XGBoost**. The goal is to classify reviews as **positive or negative** based on textual content.  

### **Steps Involved**
1. **Load & Explore the Dataset**  
2. **Preprocess Text Data** (Cleaning, Tokenization, Stopword Removal, Stemming)  
3. **Train-Test Split** (70% Train, 30% Test)  
4. **Feature Extraction** using **Bag of Words (BoW)**  
5. **Train KNN Model & Evaluate**  
6. **Train SVM, Random Forest, XGBoost on a subset (40%-50%) & Compare**  
7. **Hyperparameter Tuning using RandomizedSearchCV**  
8. **Train the Best Model on the Full Dataset**  
9. **Evaluate the Final Model (Accuracy, F1 Score, Confusion Matrix, Visualizations)**  

---

## Dataset Details
- **Dataset Source**: [IMDB Dataset](https://github.com/asRot0/machine-learning/blob/main/datasets/IMDB%20Dataset.csv)  
- **Size**: 50,000 reviews  
- **Classes**:  
  - **Positive (25,000 reviews)**  
  - **Negative (25,000 reviews)**  

Each review is labeled as **positive** or **negative**, making it a **binary classification problem**.

---

## Technologies Used
- **Python**  
- **Pandas, NumPy** (Data Handling)  
- **Scikit-Learn** (Machine Learning Models)  
- **XGBoost** (Boosting Algorithm)  
- **Seaborn, Matplotlib** (Data Visualization)  
- **NLTK, BeautifulSoup** (Text Processing)  

---

## Model Evaluation & Insights

To understand model effectiveness, we analyzed **confusion matrices** and **classification reports** for each model. Below are some key insights:

### 🔹 **K-Nearest Neighbors (KNN)**
- Performed **poorly** due to the high-dimensional sparse nature of text data.
- Struggled with decision boundaries, leading to **low accuracy**.

### 🔹 **Support Vector Machine (SVM)**
- Provided **decent performance** with good generalization.
- However, training time was relatively **slow** on a large dataset.

### 🔹 **Random Forest**
- Showed **strong results**, handling non-linear relationships well.
- Benefited from **ensemble learning**, but had slightly **higher training time**.

### 🔹 **XGBoost**
- Achieved the **best accuracy**, excelling in feature selection & boosting weak learners.
- Benefited significantly from **hyperparameter tuning**.
- Final model trained on **full dataset** after parameter optimization.

### **Visualizing Model Results**
Here’s a heatmap of the **best model’s confusion matrix**:

```python
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_final)
sns.heatmap(cm, annot=True, fmt='g', cmap="Blues")
plt.title("Best Model (XGBoost) - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
```

> **Note**: Hyperparameter tuning was performed on the best-performing model before final training.

---

## How to Run
### **1. Install Dependencies**
```bash
pip install pandas numpy scikit-learn xgboost seaborn matplotlib nltk beautifulsoup4 tqdm imbalanced-learn
