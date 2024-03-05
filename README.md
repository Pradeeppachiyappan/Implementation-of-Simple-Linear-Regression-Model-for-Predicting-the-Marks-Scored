# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

  1. Import pandas, numpy and sklearn

  2.Calculate the values for the training data set

  3.Calculate the values for the test data set

  4.Plot the graph for both the data sets and calculate for MAE, MSE and RMSE
  

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by:S.Renuga 
RegisterNumber: 212222230118
*/
```
```

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('/content/student_scores.csv')
df.head()

df.tail()

x=df.iloc[:,:-1].values
x

y=df.iloc[:,1].values
y

##  splitting train and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
y_pred

## graph plot for training data
plt.scatter(x_train,y_train,color="red")
plt.plot(x_train,regressor.predict(x_train),color="purple")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

## graph plot for test data
plt.scatter(x_test,y_test,color="red")
plt.plot(x_test,regressor.predict(x_test),color="purple")
plt.title("Hours vs Scores(Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE= ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:

## df.head()

![df head](https://github.com/RENUGASARAVANAN/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119292258/db934862-eeeb-462b-aeb7-28b869113226)



## df.tail()

![df tail](https://github.com/RENUGASARAVANAN/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119292258/42e2a9f0-638c-40c5-9ab7-1ebc5b593d69)



## ARRAY VALUE OF X

![array value of x](https://github.com/RENUGASARAVANAN/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119292258/86c89c69-0df7-499e-9805-88444385fd12)




## ARRAY VALUE OF Y

![array value of y](https://github.com/RENUGASARAVANAN/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119292258/9afae483-751b-4a77-be53-0c2cc3e73204)



## VALUES OF Y PREDICTION


![values of y prediction](https://github.com/RENUGASARAVANAN/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119292258/34ad6afe-5ee1-47bc-a293-e104dc4c0ccb)



## ARRAY VALUES OF Y TEST

![y test](https://github.com/RENUGASARAVANAN/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119292258/f7be3dfb-b4b5-44d7-9ca3-06812f202b9f)



## TRAINING SET GRAPH


![training set graph](https://github.com/RENUGASARAVANAN/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119292258/82141809-4794-44e3-a164-aa8991f22e23)




## TEST SET GRAPH

![test set graph](https://github.com/RENUGASARAVANAN/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119292258/15efd5e6-ebfd-4d31-97ae-0ad2d1295a2c)



## VALUES OF MSE,MAE AND RMSE

![values of mse,mae and rmse](https://github.com/RENUGASARAVANAN/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119292258/6793057c-b446-4e92-b11d-bdf3cba4a17b)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
