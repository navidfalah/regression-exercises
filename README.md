# 🧠 Machine Learning Exercises

Welcome to the **Machine Learning Exercises** repository! This project covers a variety of machine learning concepts and techniques, including data preprocessing, classification, regression, and model evaluation.

---

## 📂 **Project Overview**

This repository includes exercises and examples for the following topics:
- **Data Preprocessing**: Creating and manipulating datasets using Pandas.
- **Classification**: Using K-Nearest Neighbors (KNN) and Logistic Regression.
- **Regression**: Linear Regression, Ridge Regression, and Lasso Regression.
- **Model Evaluation**: Training and testing models, visualizing results.

---

## 🛠️ **Tech Stack**

- **Python**
- **Scikit-learn**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **mglearn**

---

## 📊 **Datasets**

The project uses the following datasets:
- **Breast Cancer Dataset**: For classification tasks.
- **Wave Dataset**: For regression tasks.
- **Synthetic Blobs Dataset**: For visualizing decision boundaries.

---

## 🧠 **Key Concepts**

### 1. **Data Preprocessing**
- Creating and filtering datasets using Pandas.
- Visualizing data with Matplotlib.

### 2. **Classification**
- **K-Nearest Neighbors (KNN)**: A simple and effective classification algorithm.
- **Logistic Regression**: A linear model for binary classification.

### 3. **Regression**
- **Linear Regression**: Fitting a linear model to data.
- **Ridge Regression**: Adding L2 regularization to linear regression.
- **Lasso Regression**: Adding L1 regularization to linear regression.

### 4. **Model Evaluation**
- Splitting data into training and test sets.
- Evaluating model performance using accuracy and R² scores.

---

## 🚀 **Code Highlights**

### K-Nearest Neighbors (KNN)
```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
print("Test accuracy:", knn.score(X_test, y_test))
```

### Linear Regression
```python
from sklearn.linear_model import LinearRegression
lr = LinearRegression().fit(X_train, y_train)
print("Test R² score:", lr.score(X_test, y_test))
```

### Logistic Regression
```python
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression().fit(X_train, y_train)
print("Test accuracy:", logreg.score(X_test, y_test))
```

### Ridge Regression
```python
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=0.1).fit(X_train, y_train)
print("Test R² score:", ridge.score(X_test, y_test))
```

### Lasso Regression
```python
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.01).fit(X_train, y_train)
print("Test R² score:", lasso.score(X_test, y_test))
```

### Visualizing Decision Boundaries
```python
import mglearn
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.legend()
plt.show()
```

---

## 🛠️ **Installation**

1. Clone the repository:
   ```bash
   git clone https://github.com/navidfalah/machine-learning-exercises.git
   cd machine-learning-exercises
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook:
   ```bash
   jupyter notebook exercise_machine_learning.ipynb
   ```

---

## 🤝 **Contributing**

Feel free to contribute to this project! Open an issue or submit a pull request.

---

## 📧 **Contact**

- **Name**: Navid Falah
- **GitHub**: [navidfalah](https://github.com/navidfalah)
- **Email**: navid.falah7@gmail.com
