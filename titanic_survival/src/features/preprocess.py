import pandas as pd

def preprocess(X):
    X = X.copy()
    X = X.drop(["Name", "Ticket", "Cabin"], axis=1, errors="ignore")
    X["Sex"] = X["Sex"].map({"male": 0, "female": 1})
    X["Embarked"] = X["Embarked"].map({"S": 0, "C": 1, "Q": 2})
    X.fillna(X.mean(numeric_only=True), inplace=True)
    return X