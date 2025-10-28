# Q1.
import pandas as pd  
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LogisticRegression  
from sklearn.metrics import confusion_matrix,accuracy_score  
data=pd.read_csv(r"C:/Users/OneDrive/Desktop/ML Tasks/diabetes.csv")  
feature_cols=['pregnant','insulin','bmi','age','glucose','bp']  
x=data[feature_cols]  
y=data['label']  
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)  
model=LogisticRegression(solver='lbfgs').fit(x_train,y_train)  
y_pred=model.predict(x_test)  
print(confusion_matrix(y_test,y_pred))  
print(accuracy_score(y_test, y_pred))  
                                                                   

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
