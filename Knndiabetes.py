import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

class knn:
    def __init__(self, k):
        self.k = k    
    def fit(self, xtrain, ytrain):
        self.xtrain = xtrain
        self.ytrain = ytrain   
    def predict(self, xtest):
        ypred = []
        for x in xtest:
            distances = [((x - xtrain) ** 2).sum() for xtrain in self.xtrain]
            sorted_indices = sorted(range(len(distances)), key=lambda i: distances[i])
            k_nearest_neighbors = [self.ytrain[i] for i in sorted_indices[:self.k]]
            ypred.append(max(set(k_nearest_neighbors), key=k_nearest_neighbors.count))
        return ypred
        
df = pd.read_csv("diabetes.csv")
print(df.head())
X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3, random_state=0)
model = knn(k=3)  
model.fit(X_train, Y_train)
y_pred_model = model.predict(X_test)
accuracy_model = np.sum(y_pred_model == Y_test) / len(Y_test) * 100
print("Accuracy of custom KNN:", accuracy_model)
model1 = KNeighborsClassifier(n_neighbors=3)
model1.fit(X_train, Y_train)
y_pred_model1 = model1.predict(X_test)
accuracy_model1 = np.sum(y_pred_model1 == Y_test) / len(Y_test) * 100
print("Accuracy of scikit-learn KNN:", accuracy_model1)




