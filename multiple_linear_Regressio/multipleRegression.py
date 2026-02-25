import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("multiple_linear_Regressio/dataset.csv")

X=df[['age','experience']]

Y=df['income']
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
print(x_train)
print(y_train)
model=LinearRegression()
model.fit(x_train,y_train)
print('coeficient ',model.coef_)
print(model.intercept_)
print(model.predict([[37,10]]))

predict=model.predict(x_test)
print(r2_score(predict,y_test))
print(len(x_train))
print(len(y_train))
plt.scatter(x_train["age"], y_train,marker='*')
plt.xlabel('age')
plt.ylabel('income')
plt.scatter(x_train["experience"], y_train)
plt.show()
