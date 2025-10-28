# Q1.
import pandas as pd  
data=pd.read_csv('')  
print(data)  
from pandas import DataFrame  
new_data={
    'Brand':['Ford','Ferrari','Honda'],  
    'Price':[12,15,18]  
    }  
df=DataFrame(new_data,columns=['Brand','Price'])  
export=df.to_excel(r"C:\Users\OneDrive\Desktop\filename.xlsx")


---

# Q2.
import pandas as pd  
from sklearn.model_selection import train_test_split  
from sklearn.tree import DecisionTreeClassifier  
from sklearn.metrics import confusion_matrix, accuracy_score  
data=pd.read_csv(r"C:/Users/OneDrive/Desktop/ML Tasks/cars.csv")  
print(data)  
for col in data.columns:  
    data[col], _ = pd.factorize(data[col])  
x=data.drop('class', axis=1)  
y=data['class']  
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)  
model=DecisionTreeClassifier(criterion='entropy', max_depth=3).fit(x_train,y_train)  
y_pred=model.predict(x_test)  
print(confusion_matrix(y_test,y_pred))  
print(accuracy_score(y_test,y_pred))  
