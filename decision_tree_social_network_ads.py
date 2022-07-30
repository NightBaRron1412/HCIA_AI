# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 20:08:01 2022

@author: Amir Shetaia
"""

#importing required libraries
import pandas as pd #to read the dataset
from sklearn.preprocessing import StandardScaler #for input data normalization
from sklearn.model_selection import train_test_split #to split the data
from sklearn.tree import DecisionTreeClassifier #for model creation and testing
from sklearn.metrics import confusion_matrix #to evaluate the model

#The number of iteration for calculating average model accuracy
counter = int(input("Enter the number of iterations\
                    for calculating the average model accuracy: "))

#reading dataset
dataset = pd.read_csv("Social_Network_Ads.csv") #loading data into variable
x = dataset.iloc[:, 2 : 4].values #age and salay as an input
y = dataset.iloc[:, 4].values #purchase state as an output

#scaling
sc = StandardScaler() #creating an object of the StandardScalar class
x = sc.fit_transform(x) #normalizing the dataset inputs

#splitting data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y)

AVG_ACC = 0 #A varibale to store the average accuracy

for i in range(counter):
    #learning
    classifier = DecisionTreeClassifier() #creating an object of the DecisionTreeClassifie class
    classifier.fit(x_train, y_train) #creating model

    #testing
    y_pred = classifier.predict(x_test)

    #evaluation
    cm = confusion_matrix(y_test, y_pred)

    #accuracy calculation
    acc = (cm[0][0] + cm[1][1])/(len(y_test))
    print ("The Decision Tree model accuracy in iteration #", i + 1, "is", acc * 100, "%")
    AVG_ACC += acc

print("The average accuracy of the model is", (AVG_ACC/counter)*100, "%")
