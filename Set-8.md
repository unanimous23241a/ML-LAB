# Q1.
import pandas as pd  
from sklearn.preprocessing import LabelEncoder  
from sklearn.model_selection import train_test_split  
from sklearn.naive_bayes import GaussianNB  
from sklearn.metrics import accuracy_score  
data=pd.read_csv(r"C:\Users\OneDrive\Desktop\ML Tasks\PlayTennis.csv")  
print(data)  
for col in data.columns:  
    data[col] = LabelEncoder().fit_transform(data[col])  
print(data)  
x=data.drop('play',axis=1)  
y=data['play']  
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=42)  
model=GaussianNB().fit(x_train,y_train)  
y_pred=model.predict(x_test)  
print(accuracy_score(y_test, y_pred))  
print(model.predict([[2,1,0,1]]))  
                                                                   

---

# Q2.
import pandas as pd  
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression   
from sklearn.metrics import mean_squared_error,r2_score  
data=pd.read_csv(r"C:\Users\OneDrive\Desktop\new.csv")  
data=pd.get_dummies(data, columns=['State'], drop_first=True)  
x=data.drop('Profit',axis=1)  
y=data['Profit']  
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)  
model=LinearRegression().fit(x_train,y_train)  
y_pred=model.predict(x_test)  
print(r2_score(y_test, y_pred))  
print(mean_squared_error(y_test, y_pred))  
print((mean_squared_error(y_test, y_pred))**(0.5))  
