import numpy as np
import matplotlib.pyplot as plt
import mltools as ml
import cPickle as pickle
from sklearn.neighbors import KNeighborsClassifier

X = np.genfromtxt("data/X_train.txt", delimiter = None)
Y = np.genfromtxt("data/Y_train.txt", delimiter = None)
X, Y = ml.shuffleData(X, Y)

# Xtr, Ytr = X[:20000, :], Y[:180000]
# Xval, Yval = X[180000:, :], Y[180000:]
#
# Xtr, Ytr = X[:200000, :], Y[:200000]
#
# # for i in [5,10,20,50,100,200,1000]:
# #     print 'Training a kNN classifier with k = ', i, ' neighbors'
# #     learner = ml.knn.knnClassify()
# #     learner.train(Xtr, Ytr, i)
# #     YTrainPred = learner.predict(Xtr)
# #     print 'Training Error = ', float(np.sum(YTrainPred != Ytr))/float(Xtr.shape[0])
# #     YValPred = learner.predict(Xval)
# #     print 'Validation Error = ', float(np.sum(YValPred != Yval))/float(Yval.shape[0])
#
# learner = ml.knn.knnClassify()
# learner.train(Xtr, Ytr, 100)
# YTrainPred = learner.predict(Xtr)
# print 'Training Error = ', float(np.sum(YTrainPred != Ytr))/float(Xtr.shape[0])
# with open("knn_model.mdl", "wb") as output_file:
#     pickle.dump(learner, output_file)

Xtr, Ytr = X[:1000,:], Y[:1000]
Xval, Yval = X[1000:2000,:], Y[1000:2000]

for i in [5,10,20,50,100,200,1000]:
    print 'Training a kNN classifier with k = ', i, ' neighbors'
    learner = KNeighborsClassifier(n_neighbors=i, metric='mahalanobis', metric_params={'V': np.cov(X)})
    learner.fit(Xtr, Ytr)
    YTrainPred = learner.predict(Xtr)
    print 'Training Error = ', float(np.sum(YTrainPred != Ytr))/float(Xtr.shape[0])
    YValPred = learner.predict(Xval)
    print 'Validation Error = ', float(np.sum(YValPred != Yval))/float(Yval.shape[0])
