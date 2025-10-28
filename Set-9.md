# Q1.
from sklearn.datasets import load_iris  
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.model_selection import train_test_split  
import numpy as np  
iris=load_iris()  
X_train,X_test,y_train,y_test=train_test_split(iris.data,iris.target,random_state=0)  
model=KNeighborsClassifier(n_neighbors=1).fit(X_train,y_train)  
x_new=np.array([[5, 2.9, 1, 0.2]])  
y_new_pred=model.predict(x_new)  
print("Prediction for", x_new[0], ":", iris.target_names[y_new_pred[0]])  
for i in range(len(X_test)):  
    x_sample=X_test[i].reshape(1, -1)  
    y_pred=model.predict(x_sample)  
    print(f"Actual: {y_test[i]} ({iris.target_names[y_test[i]]}),"  
          f"Predicted: {y_pred[0]} ({iris.target_names[y_pred[0]]})")  

print(f"Test set accuracy: {model.score(X_test, y_test):.2f}")  

                                                                   

---

# Q2.
#importing data
import pandas as pd  
data=pd.read_csv('')  
print(data)  

#exporting data  
from pandas import DataFrame  
new_data={  
    'Brand':['Ford','Ferrari','Honda'],  
    'Price':[12,15,18]  
    }  
df=DataFrame(new_data,columns=['Brand','Price'])  
export=df.to_excel(r"C:\Users\OneDrive\Desktop\filename.xlsx")  
 
