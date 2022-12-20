import numpy as np
import pandas as pd     
import pickle

from sklearn import tree

from sklearn.datasets import load_iris

iris=load_iris()
# print(iris)
# iris.data holds features
# iris.target holds results


x=np.array(iris.data)
y=np.array(iris.target).reshape(-1,1)

# print(x)
# print(y)

# X=np.array(dataset.drop(['play','day'],True))
# Y=np.array(dataset['play']).reshape(-1,1)

model=tree.DecisionTreeClassifier()

model=model.fit(x,y)

pickle.dump(model,open('decision_tree.model','wb'))



