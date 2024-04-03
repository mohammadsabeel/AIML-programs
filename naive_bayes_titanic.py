import pandas as pd

class NaiveBayesClassifier:
    def __init__(self,x,y):
        self.x,self.y=x,y
        self.N=len(self.x)
        self.dim=len(self.x[0])
        self.attrs=[[] for _ in range(self.dim)]
        self.output_dom={}
        self.data=[]
        
        for i in range(len(self.x)):
            for j in range(self.dim):
                if not self.x[i][j] in self.attrs[j]:
                    self.attrs[j].append(self.x[i][j])
            if not self.y[i] in self.output_dom.keys():
                self.output_dom[self.y[i]]=1
            else:
                self.output_dom[self.y[i]]+=1
            self.data.append([self.x[i],self.y[i]])
            
    def classify(self,entry):
        solve= None
        max_arg=-1
        for y in self.output_dom.keys():
            prob=self.output_dom[y]/self.N
            for i in range(self.dim):
                cases=[x for x in self.data if x[0][i]==entry[i] and x[1]==y]
                n= len(cases)
                prob*=n/self.N
            if prob>max_arg:
                max_arg=prob
                solve=y
        return solve
    
data = pd.read_csv('titanic.csv')
print(data.head())
y=list(map(lambda v : 'yes' if v==0 else 'no',data['survived'].values))
x=data[['Pclass','Age','Siblings/Spouses Aboard','Parents/Children Aboard','Fare']].values
naive_bayes=NaiveBayesClassifier(x,y)
entry_to_classify=[3,15,1,1,32.0708]
classification_result=naive_bayes.classify(entry_to_classify)
print("Classification result:",classification_result)