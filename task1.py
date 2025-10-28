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
export=df.to_excel(r"C:\Users\Mohammed Ayaz\OneDrive\Desktop\filename.xlsx")
