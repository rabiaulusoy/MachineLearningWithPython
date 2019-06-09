# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 15:20:01 2019

@author: rulusoy
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

""" DATA PREPROCESSING """
allData = pd.read_csv('data_tenis.csv')
temperature = allData.iloc[:,1:2].values #categoric data column
humidity = allData.iloc[:,2:3].values #to be predicted
outlook = allData.iloc[:,0:1].values #to be encoded
windy = allData.iloc[:,3:4].values #to be encoded
play = allData.iloc[:,4:5].values #to be encoded

#encoding what to encode(Categoric -> Numeric) and create data frames
""" as another method, encode all columns with LabelEncoder and take the part which you need to encode
from sklearn.preprocessing import LabelEncoder
allDataLabelEncoded = allData.apply(LabelEncoder().fit_transform)
labelEncoded = allDataLabelEncoded.iloc[:,-2:]
"""

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categories="auto")

outlook = ohe.fit_transform(outlook).toarray()
outlook = pd.DataFrame(data = outlook, index = range(14), columns = ['overcast','rainy','sunny'])

windy = ohe.fit_transform(windy).toarray()
windy = pd.DataFrame(data = windy[:,1:], index = range(14), columns = ['windy'])

play = ohe.fit_transform(play).toarray()
play = pd.DataFrame(data = play[:,1:], index = range(14), columns = ['play'])

temperature = pd.DataFrame(data = temperature, index = range(14), columns = ['temperature'])
humidity = pd.DataFrame(data = humidity, index = range(14), columns = ['humidity'])

#concat data frames what created
allColumnsDataFrame = pd.concat([outlook, temperature], axis = 1)
allColumnsDataFrame = pd.concat([allColumnsDataFrame, humidity], axis = 1)
allColumnsDataFrame = pd.concat([allColumnsDataFrame, windy], axis = 1)
allColumnsDataFrame = pd.concat([allColumnsDataFrame, play], axis = 1)

#concat data frames what to be splited
dataFrameExceptHumidity = pd.concat([outlook, temperature], axis = 1)
dataFrameExceptHumidity = pd.concat([dataFrameExceptHumidity, windy], axis = 1)
dataFrameExceptHumidity = pd.concat([dataFrameExceptHumidity, play], axis = 1)


""" DATA PREDICTION """
#split data as test and train
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(dataFrameExceptHumidity, humidity, test_size = 0.33, random_state = 0)

#create multiple regression and predict
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_prediction = regressor.predict(x_test)

#backward eliminatipn
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((14,1)).astype(int), values = dataFrameExceptHumidity, axis = 1)
X_l = dataFrameExceptHumidity.iloc[:,[0,1,2,3,4,5]].values
r_ols = sm.OLS(endog = humidity, exog = X_l).fit()
print(r_ols.summary())

#make elimination according to p-values of r_ols
X_l =  dataFrameExceptHumidity.iloc[:,[0,1,2,3,5]].values
r_ols = sm.OLS(endog = humidity, exog = X_l).fit()
print(r_ols.summary())

x_train, x_test, y_train, y_test = train_test_split(X_l, humidity, test_size = 0.33, random_state = 0)
regressor = LinearRegression()
regressor.fit(x_train, y_train)
y_prediction2 = regressor.predict(x_test)
#prediction improved!

#try again
X_l =  dataFrameExceptHumidity.iloc[:,[0,1,3,5]].values
r_ols = sm.OLS(endog = humidity, exog = X_l).fit()
print(r_ols.summary())

x_train, x_test, y_train, y_test = train_test_split(X_l, humidity, test_size = 0.33, random_state = 0)
regressor = LinearRegression()
regressor.fit(x_train, y_train)
y_prediction3 = regressor.predict(x_test)
#prediction result did not change

#do another prediction
X_l =  dataFrameExceptHumidity.iloc[:,[1,3,5]].values
r_ols = sm.OLS(endog = humidity, exog = X_l).fit()
print(r_ols.summary())

x_train, x_test, y_train, y_test = train_test_split(X_l, humidity, test_size = 0.33, random_state = 0)
regressor = LinearRegression()
regressor.fit(x_train, y_train)
y_prediction4 = regressor.predict(x_test)