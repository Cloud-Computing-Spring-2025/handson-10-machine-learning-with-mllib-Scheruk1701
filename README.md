# handson-10-MachineLearning-with-MLlib.

## Customer Churn Prediction with MLlib

This project uses Apache Spark MLlib to predict customer churn based on structured customer data. You will preprocess data, train classification models, perform feature selection, and tune hyperparameters using cross-validation.

Build and compare machine learning models using PySpark to predict whether a customer will churn based on their service usage and subscription features.

## Dataset

The dataset used is `customer_churn.csv`, which includes features like:

- `gender`, `SeniorCitizen`, `tenure`, `PhoneService`, `InternetService`, `MonthlyCharges`, `TotalCharges`, `Churn` (label), etc.

## Tasks

### Task 1: Data Preprocessing and Feature Engineering

**Objective:**  
Clean the dataset and prepare features for ML algorithms.

**Steps:**
1. Fill missing values in `TotalCharges` with 0.
2. Encode categorical features using `StringIndexer` and `OneHotEncoder`.
3. Assemble numeric and encoded features into a single feature vector with `VectorAssembler`.
4. Convert `Churn` label to numeric values: `Yes` -> 1.0, `No` -> 0.0.

**Code Output:**
```
+----------------------------------------+-----+
|features                                |label|
+----------------------------------------+-----+
|(8,[1,2,4,7],[30.0,53.42,1.0,1.0])      |0.0  |
|[0.0,16.0,82.08,1353.4,1.0,1.0,1.0,0.0] |0.0  |
|[0.0,69.0,37.17,2278.78,0.0,1.0,0.0,1.0]|0.0  |
|[1.0,27.0,57.86,1392.56,1.0,1.0,0.0,1.0]|0.0  |
|[1.0,2.0,72.97,158.7,0.0,0.0,1.0,0.0]   |0.0  |
+----------------------------------------+-----+
```

---

### Task 2: Train and Evaluate Logistic Regression Model

**Objective:**  
Train a logistic regression model and evaluate it using AUC (Area Under ROC Curve).

**Steps:**
1. Split dataset into training and test sets (80/20).
2. Train a logistic regression model.
3. Use `BinaryClassificationEvaluator` to evaluate performance.

**Code Output:**
```
Logistic Regression AUC: 0.7523
```

---

### Task 3: Feature Selection using Chi-Square Test

**Objective:**  
Select the top 5 most important features using Chi-Square feature selection.

**Steps:**
1. Use `ChiSqSelector` to rank and select top 5 features.
2. Print the selected feature vectors.

**Code Output:**
```
Top 5 selected features:
+--------------------------+-----+
|selectedFeatures          |label|
+--------------------------+-----+
|(5,[1,2,4],[30.0,1.0,1.0])|0.0  |
|[0.0,16.0,1.0,1.0,0.0]    |0.0  |
|[0.0,69.0,0.0,0.0,1.0]    |0.0  |
|[1.0,27.0,1.0,0.0,1.0]    |0.0  |
|[1.0,2.0,0.0,1.0,0.0]     |0.0  |
+--------------------------+-----+
```

---

### Task 4: Hyperparameter Tuning and Model Comparison

**Objective:**  
Use CrossValidator to tune models and compare their AUC performance.

**Models Used:**
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosted Trees (GBT)

**Steps:**
1. Define models and parameter grids.
2. Use `CrossValidator` for 5-fold cross-validation.
3. Evaluate and print best model results.

**Code Output:**
```
Tuning LogisticRegression...
LogisticRegression AUC: 0.7632
Tuning DecisionTree...
DecisionTree AUC: 0.7008
Tuning RandomForest...
RandomForest AUC: 0.7961
Tuning GBTClassifier...
GBTClassifier AUC: 0.7523
Best model: RandomForest with AUC = 0.7961
```

## Execution Instructions

### 1. Prerequisites

- Apache Spark installed
- Python environment with `pyspark` installed
- `customer_churn.csv` placed in the project directory

### 2. Run the Project

```bash
spark-submit customer-churn-analysis.py
```

## Code Explanation

### Task 1 - Preprocessing:
- Missing `TotalCharges` are filled with 0.
- Categorical columns (`gender`, `PhoneService`, `InternetService`) are indexed and one-hot encoded.
- Features are combined into a single vector.
- `Churn` is converted to binary label: 1.0 for churned, 0.0 for not churned.

### Task 2 - Logistic Regression:
- Trains a logistic regression model.
- Evaluates model on test data using AUC (good for imbalance).

### Task 3 - Chi-Square Feature Selection:
- Uses statistical test to select top 5 predictive features.
- Helps reduce dimensionality and improve model focus.

### Task 4 - Model Tuning:
- Cross-validation (5-fold) is used for tuning:
  - Logistic Regression: `regParam`
  - Decision Tree: `maxDepth`
  - Random Forest: `numTrees`
  - GBT: `maxIter`
- Compares all four models and prints the best based on AUC.

