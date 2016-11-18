import numpy as np
import matplotlib.pyplot as plt
import mltools as ml


# Problem 2 - Random Forests
X = np.genfromtxt("data/X_train.txt", delimiter = None)
Y = np.genfromtxt("data/Y_train.txt", delimiter = None)

Xtr, Ytr = X[:10000, :], Y[:10000]
Xval, Yval = X[10000:20000, :], Y[10000:20000]

bags = [1,5,10,25]
bagTrainError = []
bagValidationError = []
for bag in bags:
    decisionTrees = [None]*bag
    for i in range(0,bag,1):
        Xi, Yi = ml.bootstrapData(Xtr,Ytr)
        decisionTrees[i] = ml.dtree.treeClassify(Xi, Yi, maxDepth=6)

    trainingError = []
    validationError = []
    for decisionTree in decisionTrees:
        trainingError.append(decisionTree.err(Xtr, Ytr))
        validationError.append(decisionTree.err(Xval, Yval))

    bagTrainError.append(np.mean(trainingError))
    bagValidationError.append(np.mean(validationError))

plt.plot(bags, bagTrainError, 'r')
plt.plot(bags, bagValidationError, 'g')
plt.title('Error vs # of Learners in Bag')
plt.show()
