# Implementation of Simple Linear Regression Model for Predicting the Marks Scored->

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm :

1.Import pandas, numpy and sklearn.

2.Calculate the values for the training data set.

3.Calculate the values for the test data set.

4.Plot the graph for both the data sets and calculate for MAE, MSE and RMSE
  

## Program :
#### Developed by: Pradeep Raj P
#### RegisterNumber: 212222240073

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

### df.head()

![image](https://github.com/Pradeeppachiyappan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707347/6d08adae-d8cb-40a4-a03a-4641fea5bcd1)


### df.tail()

![image](https://github.com/Pradeeppachiyappan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707347/6af338f7-eb15-4aff-85d2-63a6b2008f85)

### ARRAY VALUE OF X

![image](https://github.com/Pradeeppachiyappan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707347/cfd1b324-bd01-4963-aa39-e868ce528848)

### ARRAY VALUE OF Y

![image](https://github.com/Pradeeppachiyappan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707347/4ca00856-8a72-4a5d-9f6c-43eae8014376)

### VALUES OF Y PREDICTION

![image](https://github.com/Pradeeppachiyappan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707347/212727db-68d1-4a11-8f57-3feaf13451fe)

### ARRAY VALUES OF Y TEST

![image](https://github.com/Pradeeppachiyappan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707347/06de1847-2c13-4cf3-b51e-712c677c5c1e)

### TRAINING SET GRAPH

![image](https://github.com/Pradeeppachiyappan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707347/7f7dace9-45a6-4940-adcd-33976eb87fd4)

### TEST SET GRAPH

![image](https://github.com/Pradeeppachiyappan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707347/b9e389d3-036b-46ec-91d5-d8137fadbbc4)

### VALUES OF MSE,MAE AND RMSE

![image](https://github.com/Pradeeppachiyappan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707347/eb4f49ba-63fa-4029-bbec-39c08b79f6b8)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming .
