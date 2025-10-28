# Q1.
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
                                                                   

---

# Q2.
from sklearn.cluster import KMeans  
import numpy as np  
import matplotlib.pyplot as plt  
x1 = np.array([3, 1, 1, 2, 1, 6, 6, 6, 5, 6, 7, 8, 9, 8, 9, 9, 8])  
x2 = np.array([5, 4, 6, 6, 5, 8, 6, 7, 6, 7, 1, 2, 1, 2, 3, 2, 3])  
plt.plot()  
plt.xlim([0, 10])  
plt.ylim([0, 10])  
plt.scatter(x1,x2)  
plt.show()  
plt.plot()  
X = np.array(list(zip(x1, x2))).reshape(len(x1), 2)  
colors = ['b','g','r']  
markers = ['o','v','s']  
kmeans_model = KMeans(n_clusters=3).fit(X)  
plt.plot()  
for i, l in enumerate(kmeans_model.labels_):  
    plt.plot(x1[i], x2[i], color=colors[l], marker=markers[l],ls='None')  
plt.xlim([0, 10])  
plt.ylim([0, 10])  
plt.show()  
