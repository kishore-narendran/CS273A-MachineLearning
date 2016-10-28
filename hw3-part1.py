import numpy as np
import matplotlib.pyplot as plt
import mltools as ml
import mltools.logistic2 as lc2

iris = np.genfromtxt("data/iris.txt",delimiter=None)
X, Y = iris[:,0:2], iris[:,-1]                  # get first two features & target
X, Y = ml.shuffleData(X, Y)                     # reorder randomly (important later)
X, params = ml.rescale(X)                       # works much better on rescaled data
XA, YA = X[Y<2,:], Y[Y<2]                       # get class 0 vs 1
XB, YB = X[Y>0,:], Y[Y>0]                       # get class 1 vs 2

# (a) Scatter plot of the two classes to exhibit seperability
# plt.title('Linearly Seperable Data')
# plt.scatter(XA[:, 0], XA[:, 1], c = YA)
# plt.gray()
# plt.show()
#
# plt.title('Linearly Non-seprabale Data')
# plt.scatter(XB[:, 0], XB[:, 1], c = YB)
# plt.gray()
# plt.show()

# (b) Plotting a boundary with the class data points, by modifying plotBoundary()
learner = lc2.logisticClassify2()               # Initializing the logisic classifier
learner.classes = np.unique(YA)                 # Picking uniqe values as the class labels
wts = [0.5, 1, -0.25]                           # Assigning weights
learner.theta=wts
# learner.plotBoundary(XA, YA)                    # Plotting decision boundary

# Performing above actions for the XB-YB split of the data
learner = lc2.logisticClassify2()
learner.classes = np.unique(YA)
learner.theta=wts
# learner.plotBoundary(XB, YB)

# (c) Performing prediction and finding the training data error rate
YPred = learner.predict(XA)
trainingErrorRate = np.sum((YPred - YA) ** 2)/YPred.shape[0]
print "Training Data Error Rate on Data Set A\t=\t", trainingErrorRate
YPred = learner.predict(XB)
trainingErrorRate = np.sum((YPred - YB) ** 2)/YPred.shape[0]
print "Training Data Error Rate on Data Set B\t=\t", trainingErrorRate

# (d)
ml.plotClassify2D(learner, XA, YA)
plt.show()
ml.plotClassify2D(learner, XB, YB)
plt.show()
