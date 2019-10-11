#!/usr/bin/env python
# coding: utf-8

# Importing Essential Libraries

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import joblib


# ## Importing Data


df = pd.read_csv('housing.csv')
df = df[~(df['MEDV'] >= 50.0)]


df.shape

X = df.drop('MEDV', axis = 1)
y = df['MEDV']


df.columns


# EDA

df.head()

df.isnull().sum()


df['CHAS'].value_counts()


df.info()

df.describe().T


df.hist(edgecolor = 'black', linewidth = 1.2, figsize=(20,15))
plt.show()


plt.figure(figsize=(14,8))
sns.boxplot(data = df)
plt.xticks(rotation=45, fontsize=15)
plt.show()


# Feature Selection

#  Using Correlation Matrix


plt.figure(figsize=(12,6))
sns.heatmap(df.corr(), annot=True, cmap='RdYlGn')
plt.show()


#  Selecting Highly Correlated Features

correlation = df.corr()
corr = abs(correlation['MEDV'])
target_features = corr[corr > 0.5]
target_features.drop('MEDV', inplace=True)
target_features.plot(kind = 'barh')
plt.title('Selected Featues Using Correlation', fontsize=18)
plt.show()


#  Using Lasso Regression

from sklearn.linear_model import LassoCV
model = LassoCV(cv=5)
model.fit(X, y)
print(f'Best Alpha using LassoCV: %f' % model.alpha_)
print(f'Best Score using LassoCV: %f' % model.score(X, y))
coef = pd.Series(model.coef_, index=X.columns)

print(f'Lasso Picked {str(sum(coef != 0))} featres and removed other {str(sum(coef == 0))} featres')


imp_features = coef.sort_values(ascending=False)
plt.figure(figsize=(5,5))
imp_features.plot(kind= 'barh')
plt.title('Important Features using LassoCV')
plt.show()


#  Train Test Split


X = df[['RM', 'LSTAT', 'PTRATIO', 'INDUS', 'NOX', 'TAX', 'AGE', 'CRIM']]
y = df['MEDV']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)


#  Checking Accuracy for Different Models

#  Without Scaling Data


accuracy = []
rmse = []
models = pd.Series([LinearRegression(), RandomForestRegressor(n_estimators=100), DecisionTreeRegressor(), 
                    KNeighborsRegressor(n_neighbors=3), SVR(kernel='linear', gamma='auto')])
regression = pd.Series(['Linear Reg', 'Random Forest Reg', 'Decision Tree Reg', 'KNN', 'SVR'])
for i in models:
    model = i
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy.append(model.score(X_test, y_test))
    rmse.append(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
d = {'Accuracy': accuracy, 'RMSE' : rmse}
a = pd.DataFrame(d, index=regression)
plt.figure(figsize=(10,8))
plt.subplot(2,1,1)
a['Accuracy'].plot(kind = 'barh', edgecolor = 'y')
plt.style.use('ggplot')
plt.title('Accuracy without Scaling Data', fontsize = 20)
plt.yticks(fontsize=15)
plt.subplot(2,1,2)
a['RMSE'].plot(kind = 'barh', edgecolor = 'y')
plt.style.use('ggplot')
plt.title('Accuracy without Scaling Data', fontsize = 20)
plt.yticks(fontsize=15)
plt.tight_layout(h_pad=0.8)
plt.show()


#  Using Scaled Data

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
accuracy = []
rmse = []
models = pd.Series([LinearRegression(), RandomForestRegressor(n_estimators=100), DecisionTreeRegressor(), 
                    KNeighborsRegressor(n_neighbors=3), SVR(kernel='linear', gamma='auto')])
regression = pd.Series(['Linear Reg', 'Random Forest Reg', 'Decision Tree Reg', 'KNN', 'SVR'])
for i in models:
    model = i
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy.append(model.score(X_test, y_test))
    rmse.append(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
d = {'Accuracy': accuracy, 'RMSE' : rmse}
a = pd.DataFrame(d, index=regression)
# print(a)
plt.figure(figsize=(10,8))
plt.subplot(2,1,1)
a['Accuracy'].plot(kind = 'barh')
plt.title('Accuracy with Scaled Data', fontsize= 20)
plt.yticks(fontsize=15)
plt.subplot(2,1,2)
a['RMSE'].plot(kind = 'barh')
plt.title('RMSe with Scaled Data', fontsize= 20)
plt.yticks(fontsize=15)
plt.tight_layout(h_pad=0.8)
plt.show()


#  Resacling and Normalizing Data in Pipeline

steps = [('scaler', StandardScaler()), ('RFR', RandomForestRegressor(n_estimators=100))]
pipeline = Pipeline(steps)
pipeline.fit(X_train, y_train)
# y_pred = pipeline.predict(X_test)
# print(pipeline.score(X_test, y_test))


#  Saving Model

from joblib import dump, load
dump(pipeline, 'price_pred_model.pkl')
a = load('price_pred_model.pkl')
print(a.score(X_test, y_test))

