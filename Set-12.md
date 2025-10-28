# Q1.
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
colors = ['b', 'g', 'r']  
markers = ['o', 'v', 's']  
kmeans_model = KMeans(n_clusters=3).fit(X)  
plt.plot()  
for i, l in enumerate(kmeans_model.labels_):  
    plt.plot(x1[i], x2[i], color=colors[l], marker=markers[l],ls='None')  
plt.xlim([0, 10])  
plt.ylim([0, 10])  
plt.show()   
                                                                   

---

# Q2.
import pandas as pd  
from sklearn.preprocessing import StandardScaler,Binarizer,MinMaxScaler  
df=pd.read_excel(r"C:\Users\Mohammed Ayaz\Downloads\diabetes_sample.xlsx",names=['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi','age', 'class'])  
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
                                                                   
