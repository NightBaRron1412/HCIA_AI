# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 21:48:40 2022

@author: amoor
"""

#Linear regression
import pandas as pd
import matplotlib.pyplot as plt

#Read Dataset
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:, [0]].values #x must be a matrix
y = dataset.iloc[:,1].values

#Split dataset into training and testing data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.30)

#Learning Linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train) #Model is created

#Testig
y_pred=regressor.predict(x_test)

#Mean square error
from sklearn.metrics import mean_squared_error
error=mean_squared_error(y_test, y_pred)

#visualization of the training set
plt.scatter(x_train, y_train, color = "red")
plt.plot(x_train, regressor.predict(x_train), color = "blue")
plt.title("Salary Vs Experience (Training Set)")
plt.xlabel("Years of Experinece")
plt.ylabel("Salary")
plt.show()

#visualization of the test set
plt.scatter(x_test, y_test, color = "red")
plt.plot(x_train, regressor.predict(x_train), color = "blue")
plt.title("Salary Vs Experience (Test Set)")
plt.xlabel("Years of Experinece")
plt.ylabel("Salary")
plt.show()

sal = regressor.predict([[4.2]])