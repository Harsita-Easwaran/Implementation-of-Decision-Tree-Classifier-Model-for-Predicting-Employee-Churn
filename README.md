# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Harsita Easwaran
RegisterNumber:  24013629
*/
```
import pandas as pd
data=pd.read_csv("Employee.csv")
print("data.head")
data.head()
print("data.info():")
data.info()
print("isnull() and sum():")
data.isnull().sum()
print("data value counts():")
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
print("data.head()for salary")
data["salary"]=le.fit_transform(data["salary"])
data.head()
print("x.head():")
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours"]]
x.head()     #no departments and no left
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
print("Accuracy value:")
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plot_tree(dt,feature_names=x.columns,class_names=['salary','left'],filled=True)
plt.show()

## Output:
![decision tree classifier model](sam.png)
![Screenshot 2024-11-28 233514](https://github.com/user-attachments/assets/05619003-e2ad-4de6-99a9-3cd2c49f5041)
![Screenshot 2024-11-28 233537](https://github.com/user-attachments/assets/f692a7a2-cced-4442-9dd2-150509f25592)
![Screenshot 2024-11-28 233546](https://github.com/user-attachments/assets/67a029df-39e5-453b-bfa6-5e79492289af)
![Screenshot 2024-11-28 233554](https://github.com/user-attachments/assets/e2b88ccf-dfc5-4867-b3f1-b94ab46e1e1e)
![Screenshot 2024-11-28 233642](https://github.com/user-attachments/assets/223c15b9-a6a8-4316-bfde-ae16d1aea882)
![Screenshot 2024-11-28 233652](https://github.com/user-attachments/assets/1bbf6080-5cf5-4a27-9d75-bc0794fe3dea)
![Screenshot 2024-11-28 233659](https://github.com/user-attachments/assets/29745370-5f30-49de-8466-2688225c47dd)
![Screenshot 2024-11-28 233712](https://github.com/user-attachments/assets/13426c29-56ad-48fe-b6c2-6067dce10a30)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
