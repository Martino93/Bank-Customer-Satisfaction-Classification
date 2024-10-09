**The goal** of this project is to find the best classification model that can predict whether a customer is satisfied with their bank. 

In this case, we use the [Santander dataset](https://www.kaggle.com/competitions/santander-customer-satisfaction) from Kaggle.

Some limitations of this include:
- A skewed dataset towward satisfied customers (96.04%)
- Columns with constant values (34 columns)
- Number of features (308)
- Lack of metadata

The main file showing the surface analysis is **main.ipynb**.

Most of the python code is found in PHYS247.py which has custom dependencies used in the primary analysis.

### Methods Used
- Data cleaning
- Feature scaling
- Dmensionality reduction
- Model training (logistic regression, random forest, decision tree, xgboost, neural network)
- Model evaluation
- Hyperparameter tuning
