# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 14:08:48 2019

@author: rulusoy
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
import statsmodels.formula.api as sm

""" DATA PREPROCESSING """
#data load
allData = pd.read_csv('maaslar_yeni.csv')
#column slice

#p-values of x2 ve x3 columns higher than we expect
#we can compare r-square values by adding or removing these columns
#x = allData.iloc[:,2:3]
x = allData.iloc[:,2:5]
y = allData.iloc[:,5:]

#numpy array conversion
X = x.values
Y = y.values
#print(allData.corr())


""" LINEAR REGRESSION """
from sklearn.linear_model import LinearRegression
linRegressor = LinearRegression()
linRegressor.fit(X, Y)

print("Linear OLS")
model = sm.OLS(linRegressor.predict(X), X)
print(model.fit().summary())


""" POLYNOMIAL REGRESSION """
from sklearn.preprocessing import PolynomialFeatures
polynomialRegressor = PolynomialFeatures(degree = 4)
x_poly = polynomialRegressor.fit_transform(X)

linearRegressor = LinearRegression()
linearRegressor.fit(x_poly,y)

print("Polynomial OLS")
modelPoly = sm.OLS(linearRegressor.predict(polynomialRegressor.fit_transform(X)),X)
print(modelPoly.fit().summary())


""" Standart Scaler """
from sklearn.preprocessing import StandardScaler
scalerX = StandardScaler()
x_scaled = scalerX.fit_transform(X)
scalerY = StandardScaler()
y_scaled = scalerY.fit_transform(Y)


""" SVR """
from sklearn.svm import SVR
svrRegressior = SVR(kernel='rbf')
svrRegressior.fit(x_scaled,y_scaled)

print("SVR OLS")
modelSVR = sm.OLS(svrRegressior.predict(x_scaled),x_scaled)
print(modelSVR.fit().summary()) 


""" DT """
from sklearn.tree import DecisionTreeRegressor
decisionTreeRegressor = DecisionTreeRegressor(random_state = 0)
decisionTreeRegressor.fit(X, Y)

print("Decision Tree OLS")
modelDT = sm.OLS(decisionTreeRegressor.predict(X),X)
print(modelDT.fit().summary())


""" RANDOM FOREST """
from sklearn.ensemble import RandomForestRegressor
randomForestRegressor = RandomForestRegressor(n_estimators=10, random_state = 0)
randomForestRegressor.fit(X,Y)

print("Random Forest OLS")
modelRF = sm.OLS(randomForestRegressor.predict(X),X)
print(modelRF.fit().summary())