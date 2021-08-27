from sklearn import datasets
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
iris=datasets.load_iris()

print(list(iris.keys())) #structure 
print(iris["data"])
print(iris["target"])
print(iris["DESCR"])
print(iris["feature_names"])
#petal width #isis virginica
x=iris['data'][:,3:]
print(x)
y=(iris["target"]==2).astype(np.int)
print(y)
#train
clf=LogisticRegression()
clf.fit(x,y)
example=clf.predict(([[2.4]]))
print(example)

#ploting 
X_new=np.linspace(0,3,1000).reshape(-1,1)
print(X_new)
Y_prob=clf.predict_proba(X_new)
plt.plot(X_new,Y_prob[:,1])
plt.xlabel("Petal Width")
plt.show()

