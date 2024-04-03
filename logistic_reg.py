import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def sigmoid(z):
	return 1.0/(1.0+np.exp(-z))
	
def cost_function(h,y):
	return (-y * np.log(h) - (1-y)*np.log(1-h)).mean()
	
def gradient(X,h,y):
	return np.dot(X.T,(h-y))/y.shape[0]
	
def logistic_regression(X,y,num_iterations,learning_rate):
	weights=np.zeros(X.shape[1])
	for i in range(num_iterations):
		z=np.dot(X,weights)
		h=sigmoid(z)
		gradient_val=gradient(X,h,y)
		weights -= learning_rate * gradient_val
	return weights
	
diabetes = pd.read_csv('diabetes.csv')
print(diabetes.head())
X= diabetes.drop('Outcome',axis=1).values
y=diabetes['Outcome'].values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=9)

sc=StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std =sc.transform(X_test)

weights_custom = logistic_regression(X_train_std,y_train,num_iterations=200,learning_rate=0.1)

z_custom=np.dot(X_test_std,weights_custom)
y_pred_custom=sigmoid(z_custom)>0.5

accuracy_custom=np.mean(y_pred_custom==y_test)
print('Custom Model Accuracy:',accuracy_custom) 
 

lr_model=LogisticRegression()
lr_model.fit(X_train_std,y_train)
y_pred_lr = lr_model.predict(X_test_std)

accuracy_lr = accuracy_score(y_test,y_pred_lr)
print('Scikit Accuracy:',accuracy_lr)
