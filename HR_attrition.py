# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 10:45:30 2021

@author: Karthik C V
"""


###############################################
###         Import the libraries          #####
###############################################
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import statsmodels.formula.api as sm
import matplotlib.pyplot as plt



employee_data=pd.read_csv(r"F:\Machine_Learning\EVEN_2021\logistic regression\demo\EMPLOYEE_ATTRITION.csv")
employee_data.dtypes

employee_data.isnull().sum()


#to find the number of posiive and negative observations in the response variable
employee_data['Attrition'].value_counts()
sns.countplot(employee_data['Attrition'])

#creating a list of all independent variables 
X_features = list(employee_data)
X_features.remove('Attrition')
X_features

#Encoding Categorical Features
employee_data_df = pd.get_dummies(employee_data[X_features],drop_first = True)
Attrition_dm=pd.get_dummies(employee_data['Attrition'])

#defining X and Y features for model building
import statsmodels.api as sm 
Y = Attrition_dm['No']
X = employee_data_df


#splitting the dataset into training and testing set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size = 0.7)

#building the logistic regression model
from sklearn.linear_model import LogisticRegression
model= LogisticRegression()
model.fit(X_train,Y_train)
print(model.coef_, model.intercept_)

###################################
Y_predict=model.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(Y_test,Y_predict)

from sklearn.metrics import confusion_matrix
confusion_matrix(Y_test,Y_predict)
pd.crosstab(Y_test,Y_predict)


tn,fp,fn,tp=confusion_matrix(Y_test,Y_predict).ravel()
print("True Negatives:  ",tn)
print("False Negatives:  ",fn)
print("True positives:  ",tp)
print("False positives:  ",fp)


############################################################


X_updated=employee_data_df['Age','DistanceFromHome','Gender_Male','WorkLifeBalance']
X_train,X_test,Y_train,Y_test = train_test_split(X_updated,Y,train_size = 0.7)
from sklearn.linear_model import LogisticRegression
model2= LogisticRegression()
model2.fit(X_train,Y_train)

#uploading the model to heroku
import pickle
#save the model to disk in wordbytes format
pickle.dump(model2, open('model.pkl','wb'))
