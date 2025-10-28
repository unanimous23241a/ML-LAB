# Q1.
import pandas as pd  
from sklearn.preprocessing import StandardScaler,Binarizer,MinMaxScaler  
df=pd.read_excel(r"C:\Users\Downloads\diabetes_sample.xlsx",names=['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi','age', 'class'])  
array=df.values  
x=array[:,:8]  
y=array[:,8]  
#StandardScaler  
x_scaled=StandardScaler().fit_transform(x)  
print(x_scaled)  
#Binarizer  
x_binary=Binarizer(threshold=0.0).fit_transform(x)  
print(x_binary)  
#Min-Max Scaler  
x_minmax=MinMaxScaler((0,1)).fit_transform(x)  
print(x_minmax)  
                                                                   

---

# Q2.
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
