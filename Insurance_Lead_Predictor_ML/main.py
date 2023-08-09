import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.linear_model import RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier


# Data Import
df = pd.read_csv('Health Insurance Lead Prediction Raw Data.csv')


# EDA & Preprocessing
print(df.info())
print(df.isnull().sum())

for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    df[col] = df[col].astype('float')

print(df, df.info())

df = df.dropna()
print(df.info(), df.isnull().sum())

columns_to_drop = ['ID', 'City_Code']

df = df.drop(columns_to_drop, axis= 1)
print(df.columns)

# Train Test Split
features = df.drop('Response', axis= 1)
target = df['Response']

print(features.info(), target.info())

X_train, X_test, Y_train, Y_test = train_test_split(features, target, random_state= 42, test_size= 0.25)


# Model Training & Voting
models = [VotingClassifier([['rc', RidgeClassifier()], ['dtc', DecisionTreeClassifier()], ['rfc', RandomForestClassifier()], ['abc', AdaBoostClassifier()], ['xgbc', XGBClassifier()]]),
          RidgeClassifier(),
          DecisionTreeClassifier(),
          RandomForestClassifier(),
          AdaBoostClassifier(),
          XGBClassifier()]

for m in models:
    print(m)
    m.fit(X_train, Y_train)

    pred_train = m.predict(X_train)
    print(f'Training Accuracy is : {accuracy_score(Y_train, pred_train)}')

    pred_test = m.predict(X_test)
    print(f'Test Accuracy is : {accuracy_score(Y_test, pred_test)}\n')

