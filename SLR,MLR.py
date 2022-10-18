# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 12:06:28 2022

@author: asimp
"""
#importing the datasets

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

dataset=pd.read_csv(r'D:\DataSets/kc_house_data.csv')
space=dataset['sqft_living']
price=dataset['price']

x=np.array(space).reshape(-1,1)
y=np.array(price)

"""SIMPLE LINEAR REGRESSION"""

#splitting the data into training and testing phase
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
#fitting simple linear regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

#predicting The Prices
predicts=regressor.predict(x_test)

#visualizaing the training and testing pbase
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Visuals for Training Data")
plt.xlabel("space")
plt.ylabel("Price")
plt.show()

#Visuals for Test Results
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Visuals for Test Data")
plt.xlabel("space")
plt.ylabel("price")
plt.show()

"""MULTIPLE LINEAR REGRESSION"""

dataset.head()

#checking if thr value is missing
print(dataset.isnull().any())

#checking for categorical data
print(dataset.dtypes)

#dropping the id and date column
dataset=dataset.drop(['id','date'],axis=1)
dataset.head()

#understanding the distribution with seaborn
with sns.plotting_context("notebook",font_scale=3):
    g=sns.pairplot(dataset[['sqft_lot','sqft_above','price','sqft_living','bedrooms']],
                   hue='bedrooms',palette='tab20',size=8) 
    g.set(xticklabels=[]);
    
#seperating independent and Dependent Variable

x=dataset.iloc[:,1:].values
y=dataset.iloc[:,0].values

#splitting dataset into training and testing phase
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor1=LinearRegression()
regressor1.fit(x_train,y_train)

#Precticting the test set Results
 y_predicts=regressor1.predict(x_test)

#Backward Elimination
import statsmodels.formula.api as sm
def backwardElimination(x,SL):
    numVars=len(x[0])
    temp=np.zeros((21613,19)).astype(int)
    for i in range(0,numVars):
        regressor1_OLS=sm.OLS(y,x).fit()
        maxVar=max(regressor1_OLS.pvalues).astype(float)
        adjR_before=regressor1_OLS.rsquared_adj.astype(float)
        if maxVar>SL:
            for j in range(0,numVars-i):
                if (regressor1_OLS.pvalues[j].astype(float)==maxVar):
                    
                    temp[:,j]=x[:,j]
                    x=np.delete(x,j,1)
                    tmp_regressor=sm.OLS(y,x).fit()
                    ajR_after=tmp_regressor.rquared_aj.astype(float)
                    if(adjR_before>= adjR_after):
                        x_rollback=np.hstack((x,temp[:,[0,j]]))
                        x_rollback=np.delete(x_rollback,j,1)
                        print(regressor1_OLS.summary())
                        return X_rollback
                    else:
                        continue
                    regressor1_OLS.summary()
                    return x
                
                SL =0.05
                x_opt=x[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]]
                x_Modeled=backwardElimination(x_opt,SL)