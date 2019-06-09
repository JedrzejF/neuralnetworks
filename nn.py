from sklearn.neural_network import MLPClassifier #sieci nueronowe
import pandas as pd
import numpy as np
import statsmodels.api as sm

train = pd.read_csv('./train/train.tsv', sep='\t', header=0)
columns = train.columns.tolist()
columns.remove('Survived')
test = pd.read_csv('./test-A/in.tsv', sep='\t', header=None, names=columns)
dev = pd.read_csv('./dev-0/in.tsv', sep='\t', header=None, names=columns)

def prep(dataset):
    dataset = dataset.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    dataset['Sex'] = dataset['Sex'].map({'female':0, 'male':1})
    dataset['Embarked'] = dataset['Embarked'].map({'S':0, 'C':1, 'Q':2})                                                                                                                   
    dataset['Embarked'].fillna(dataset['Embarked'].mean(), inplace=True)
    linmodel = sm.OLS(dataset['Age'][dataset.Age.notnull()], dataset[['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']][dataset.Age.notnull()]).fit()
    dataset['Age'][dataset.Age.isnull()] = linmodel.predict(dataset[['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']][dataset.Age.isnull()])
    return dataset

train = prep(train)
test = prep(test)
dev = prep(dev)

y = train['Survived']
X = train.drop(['Survived'], axis=1)
network = MLPClassifier()
model = network.fit(X, y)

test_pred = network.predict(test)
dev_pred = network.predict(dev)

with open('./test-A/out.tsv', 'w') as output:
    for i in test_pred:
        output.write(str(i) + '\n')

with open('./dev-0/out.tsv', 'w') as output2:
    for i in dev_pred:
        output2.write(str(i) + '\n')