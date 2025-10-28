# Q1.
import pandas as pd  
import matplotlib.pyplot as plt  
data = [['E001', 'M', 34, 123, 'Normal', 350],  
['E002', 'F', 40, 114, 'Overweight', 450],  
['E003', 'F', 37, 135, 'Obesity', 169],  
['E004', 'M', 30, 139, 'Underweight', 189],  
['E005', 'F', 44, 117, 'Underweight', 183],  
['E006', 'M', 36, 121, 'Normal', 80],  
['E007', 'M', 32, 133, 'Obesity', 166],  
['E008', 'F', 26, 140, 'Normal', 120],  
['E009', 'M', 32, 133, 'Normal', 75],  
['E010', 'M', 36, 133, 'Underweight', 40]]  
df=pd.DataFrame(data,columns=['EMPID', 'Gender','Age', 'Sales','BMI', 'Income'])  
#Histogram  
df.hist()  
plt.show()  
#BarChart  
df.plot.bar()  
plt.bar(df['Age'],df['Sales'])  
plt.xlabel('Age')  
plt.ylabel('Sales')  
plt.show()  
#BoxPlot  
df.plot.box()  
plt.boxplot(df['Income'])  
plt.show()  
#PieChart  
plt.pie(df['Age'], labels = {"A", "B", "C", "D", "E", "F","G", "H", "I", "J"},autopct ='% 1.1f %%',shadow = False)  
plt.show()  
#ScatterPlot  
plt.scatter(df['Income'],df['Sales'])  
plt.show()    
                                                                   

---

# Q2.
To be added 
