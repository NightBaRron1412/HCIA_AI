# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 20:30:19 2022

@author: Amir Shetaia
"""

#importing required libraries
import pandas as pd #to read the dataset
from sklearn.preprocessing import StandardScaler #for input data normalization
from sklearn.model_selection import train_test_split #to split the data
from sklearn.ensemble import RandomForestClassifier  #for model creation and testing
from sklearn.metrics import confusion_matrix #to evaluate the model


#reading dataset
dataset = pd.read_csv("Social_Network_Ads.csv") #loading data into variable
x = dataset.iloc[:, 2 : 4].values #age and salay as an input
y = dataset.iloc[:, 4].values #purchase state as an output

#scaling
sc = StandardScaler() #creating an object of the StandardScalar class
x = sc.fit_transform(x) #normalizing the dataset inputs

#splitting data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y)



#learning
classifier = RandomForestClassifier() #creating an object of the RandomForestClassifier class
classifier.fit(x_train, y_train) #creating model

#testing
y_pred = classifier.predict(x_test)

#evaluation
cm = confusion_matrix(y_test, y_pred)

#accuracy calculation
acc = (cm[0][0] + cm[1][1])/(len(y_test))
print ("The Random Forest model accuracy in iteration #", i + 1, "is", acc * 100, "%")
