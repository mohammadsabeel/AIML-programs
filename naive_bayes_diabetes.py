import pandas as pd

class NaiveBayesClassifier:    
    def __init__(self, X, y):      
        self.X, self.y = X, y         
        self.N = len(self.X) 
        self.dim = len(self.X[0]) 
        self.attrs = [[] for _ in range(self.dim)] 
        self.output_dom = {} 
        self.data = []           
        for i in range(len(self.X)):
            for j in range(self.dim):        
                if not self.X[i][j] in self.attrs[j]:
                    self.attrs[j].append(self.X[i][j])
            if not self.y[i] in self.output_dom.keys():
                self.output_dom[self.y[i]] = 1         
            else:
                self.output_dom[self.y[i]] += 1          
            self.data.append([self.X[i], self.y[i]])   

    def classify(self, entry):
        solve = None 
        max_arg = -1 
        for y in self.output_dom.keys():
            prob = self.output_dom[y]/self.N
            for i in range(self.dim):
                cases = [x for x in self.data if x[0][i] == entry[i] and x[1] == y] 
                n = len(cases)
                prob *= n/self.N      
            if prob > max_arg:
                max_arg = prob
                solve = y
        return solve      
        
data = pd.read_csv('diabetes.csv')
print(data.head())
y = list(map(lambda v: 'yes' if v == 1 else 'no', data['Outcome'].values))
X = data[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']].values 
naive_bayes = NaiveBayesClassifier(X, y)
entry_to_classify = [6, 148, 72, 35, 0, 33.6, 0.627, 50]
classification_result = naive_bayes.classify(entry_to_classify)
print("Classification result:", classification_result)

    