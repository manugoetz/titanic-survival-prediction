# Titanic Survival Prediction

**Kaggle Challenge:** [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic)  
**Author:** Manuel Götz 

Predicting passenger survival on the Titanic using machine learning. This project includes **data exploration, feature engineering, modeling**, and a **Streamlit app** for interactive predictions.

---

## Table of Contents

1. [Motivation](#motivation)  
2. [Dataset](#dataset)  
3. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)  
4. [Feature Engineering](#feature-engineering)  
5. [Modeling](#modeling)  
6. [Streamlit App](#streamlit-app)  
7. [How to Run](#how-to-run)  
8. [Future Improvements](#future-improvements)

---

## Motivation

The Titanic dataset is a classic machine learning problem. The goal is to **predict survival** based on passenger attributes such as age, sex, ticket class, and family information. This project demonstrates:

- End-to-end ML workflow  
- Data preprocessing and feature engineering  
- Modeling and evaluation  
- Deployment of an interactive app

---

## Dataset

The Titanic dataset contains passenger information used to predict survival. The following table summarizes each variable.

**Data Dictionary**<br>

| **Variable**  | **Definition**                    | **Key**                                                          |
|---------------|-----------------------------------|------------------------------------------------------------------|
| survival      | Survival                          | 0 = No, 1 = Yes                                                  |
| pclass        | Ticket class                      | 1 = 1st, 2 = 2nd, 3 = 3rd                                        |
| sex           | Sex                               |                                                                  |
| Age           | Age in years                      |                                                                  |
| sibsp         | # of siblings / spouses aboard    |                                                                  |
| parch         | # of parents / children aboard    |                                                                  |
| ticket        | Ticket number                     |                                                                  |
| fare          | Passenger fare                    |                                                                  |
| cabin         | Cabin number                      |                                                                  |
| embarked      | Port of Embarkation               | C = Cherbourg (FRA), Q = Queenstown (IRL), S = Southampton (ENG) |

**Variable Notes**<br>
**pclass**: A proxy for socio-economic status (SES)<br>
1st = Upper<br>
2nd = Middle<br>
3rd = Lower<br>

**age**: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5<br>

**sibsp**: The dataset defines family relations in this way...<br>
Sibling = brother, sister, stepbrother, stepsister<br>
Spouse = husband, wife (mistresses and fiancés were ignored)<br>

**parch**: The dataset defines family relations in this way...<br>
Parent = mother, father<br>
Child = daughter, son, stepdaughter, stepson<br>
Some children travelled only with a nanny, therefore parch=0 for them.<br>

> **Note:** EDA is performed **only on the training set** to prevent data leakage. The test set is reserved for final predictions.

---

## Exploratory Data Analysis (EDA)

EDA is conducted on the **training data** to explore relationships between features and survival:

- Distribution of numerical features (Age, Fare)  
- Survival rates by categorical features (Sex, Pclass, Embarked)  
- Missing value analysis  
- Insights guiding feature engineering

---

## Feature Engineering

New features are created based on EDA insights:

- `FamilySize` = `SibSp + Parch + 1`  
- `Title` extracted from passenger names  
- Imputation of missing Age values using a **GroupMedianImputer** based on Pclass and Sex  

All transformations are implemented in **modular scripts** and applied consistently to train and test data.

---

## Modeling

- Multiple models trained (Logistic Regression, Random Forest, Gradient Boosting)  
- Preprocessing pipelines handle numerical and categorical features  
- Cross-validation used for performance assessment  
- Best model selected based on accuracy and F1-score

---

## Streamlit App

An interactive app allows users to **input passenger information** and receive a **predicted survival probability**:

- Input features: Age, Sex, Pclass, SibSp, Parch, Fare, Embarked  
- Real-time predictions based on the trained model  

App can be run locally or deployed online.  
https://titanic-survival-prediction-manugoetz.streamlit.app/

---

## How to Run

### Clone the repo
```bash
git clone https://github.com/yourusername/titanic-survival-prediction.git
cd titanic-survival-prediction
```

### Setup environment
```bash
conda create -n titanic python=3.10
conda activate titanic
pip install -r requirements.txt
```

### Run notebooks
```bash
jupyter notebook
```

### Run Streamlit app
```bash
streamlit run app/titanic_app.py
```

## Future Improvements

- Hyperparameter tuning and model ensembling
- Feature importance analysis and error analysis
- Deploy Streamlit app online
- Extend to more complex or real-world datasets