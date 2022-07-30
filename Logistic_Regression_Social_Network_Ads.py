# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 18:53:40 2022

@author: Amir Shetaia
"""

#Network Social Ads Model

#importing libraries
import pandas as pd

#reading dataset
dataset = pd.read_csv("Social_Network_Ads.csv")
x = dataset.iloc[:,2:4].values
y = dataset.iloc[:, 4].values

#scaling / normalization
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)

#splitting dataset into training and testing data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .30)

#training
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(x_train, y_train) #creating model

#testing
y_pred=classifier.predict(x_test)

#evaluation
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#accuracy calculation
acc = (cm[0][0]+cm[1][1])/(len(y_test))
print("The logistic regression model accuracy is:", acc*100, "%")