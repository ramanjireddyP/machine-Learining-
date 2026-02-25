import numpy as np 
import pandas as pd
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
df = pd.read_csv("test.csv")
print(df.isnull().sum())
df_test=pd.read_csv('train.csv')
df_test.dropna(inplace=True)
print(df_test.isnull().sum())
print(df_test)
print(df.info())
print(df[['x']])
model=LinearRegression()
X=df[['x']]
Y=df['y']
x_test=df_test[['x']]
y_test=df_test['y']

model.fit(X,Y)
predict_1=model.predict(x_test)
predict_2=model.predict(X)
print(r2_score(y_test,predict_1))
print(model.coef_)
print(model.intercept_)
plt.scatter(X,Y,marker='*',color='black')
plt.xlabel('x')
plt.ylabel('y')

plt.plot(X,predict_2,color='red')

print(model.predict([[71]]))
print(df)
print(df.iloc[295,1])
plt.show()

