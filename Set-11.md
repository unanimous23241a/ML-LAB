# Q1.
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

                                                                   

---

# Q2.
import pandas as pd  
from sklearn.model_selection import train_test_split  
from sklearn.ensemble import GradientBoostingRegressor  
from sklearn.metrics import r2_score, mean_squared_error  
data=pd.read_csv(r"C:\Users\OneDrive\Desktop\ML Tasks\diabetes.csv")  
x=data.drop("label",axis=1)  
y=data["label"]  
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)  
model=GradientBoostingRegressor().fit(x_train,y_train)  
y_pred=model.predict(x_test)  
print(r2_score(y_test, y_pred))  
print(mean_squared_error(y_test,y_pred))  
