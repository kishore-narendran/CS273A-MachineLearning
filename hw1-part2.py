import numpy as np
import matplotlib.pyplot as plt
import mltools as ml
iris = np.genfromtxt("data/iris.txt",delimiter=None)
Y = iris[:,-1] # target value is the last column
X = iris[:,0:-1] # features are the other columns

X,Y = ml.shuffleData(X,Y); # shuffle data randomly
Xtr,Xte,Ytr,Yte = ml.splitData(X,Y, 0.75); # split data into 75/25 train/test

# (a) Plotting classification boundary for two features in the iris dataset
K = [1, 5, 10, 50]
for i in K:
    knn = ml.knn.knnClassify()
    knn.train(Xtr[:,0:2], Ytr, i)
    ml.plotClassify2D(knn, Xtr[:,0:2], Ytr)
    plt.show()

# (b) Computing the error rate for the training data and testing data once having
#    trained a kNN classifier, and printing the error rate vs k graph
K = [1, 2, 5, 10, 50, 100, 200]
errTrain = []
errTest = []
for i,k in enumerate(K):
    learner = ml.knn.knnClassify()
    learner.train(Xtr[:,0:2], Ytr, k)
    YTrainPred = learner.predict(Xtr[:,0:2])
    errTrain.append(float(np.sum(YTrainPred != Ytr))/float(Xtr.shape[0]))

    YTestPred = learner.predict(Xte[:,0:2])
    errTest.append(float(np.sum(YTestPred != Yte))/float(Xte.shape[0]))

plt.semilogx(K, errTrain, color='r')
plt.semilogx(K, errTest, color='g')
plt.xticks(K, K)
plt.show()
