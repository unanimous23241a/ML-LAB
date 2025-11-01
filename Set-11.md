# Q1. PCA Method
from numpy import array,mean,cov  
from numpy.linalg import eig   
#Creating Array  
A=array([[1,2],[3,4],[5,6]])  
print(A)  
#Calculate MEan  
M=mean(A.T,axis=1)  
print(M)  
#Reducing Each Feature  
C=A-M  
print(C)  
#Calculate CoVarinace  
V=cov(C.T)  
print(V)  
#Calculate Eigen Val & Vectors  
values,vectors=eig(V)  
print(values,vectors)  
#Calculate PCA  
P=vectors.T.dot(C.T)  
print(P)  

                                                                   

----------------------------------------
----------------------------------------
----------------------------------------



# Q2.  Bssoting ensemble methodimport pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error

from sklearn.ensemble import GradientBoostingRegressor
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label'] # load dataset

pima = pd.read_csv("D:\soujanya\Machine Learning 22-23 I Sem\Machine Learning 2022-23 II sem\ML LAB\Datasets/diabetes.csv", header=None, names=col_names)

print(pima)
feature_cols = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp']
X = pima[feature_cols] # Features
y = pima.label # Target variable

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20) # initializing the boosting

model = GradientBoostingRegressor()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import r2_score
print('score==',r2_score(y_test,y_pred))
print('mean_sqrd_error is==',mean_squared_error(y_test,y_pred))
print('root_mean_squared error of is==',np.sqrt(mean_squared_error(y_test,y_pred)))
