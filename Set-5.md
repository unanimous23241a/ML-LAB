# Q1.
import numpy as np  
import matplotlib.pyplot as plt  
#b1=∑((x−xˉ)*(y−yˉ))/∑(x−xˉ)^2  
#b0​=yˉ​−b1​*xˉ  
def coef(x,y):  
    x_bar,y_bar=np.mean(x),np.mean(y)  
    b1=np.sum((x-x_bar)*(y-y_bar))/np.sum((x-x_bar)**2)  
    b0=y_bar-b1*x_bar  
    return b0,b1  

def plot(x,y,b):  
    plt.scatter(x,y)  
    plt.plot(x,b[0]+b[1]*x)  
    plt.xlabel('x')  
    plt.ylabel('y')  
    plt.show()  

x=np.array([0,1,2,3,4,5,6,7,8,9])  
y=np.array([1,3,2,5,7,8,8,9,10,12])  
b=coef(x,y)  
print(b[0],b[1])  
plot(x,y,b)  


                                                                   

---

# Q2.
import pandas as pd  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.model_selection import train_test_split, cross_val_score  
from sklearn.metrics import classification_report, confusion_matrix  
data=pd.read_csv(r"C:\Users\OneDrive\Desktop\ML Tasks\diabetes.csv")  
print(data)  
x=data.drop("label",axis=1)  
y=data["label"]  
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)  
model=RandomForestClassifier().fit(x_train,y_train)  
y_pred=model.predict(x_test)  
cv_score=cross_val_score(model,x,y,cv=5,scoring='roc_auc')  
print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))  
print(cv_score)  
print(cv_score.mean())   
