import numpy as np
import matplotlib.pyplot as plt
import mltools as ml

# Best noted performance is from depth = 7, minLeaves = 4

X = np.genfromtxt("data/X_train.txt", delimiter = None)
Y = np.genfromtxt("data/Y_train.txt", delimiter = None)

Xtr, Ytr = X[:10000, :], Y[:10000]
Xval, Yval = X[10000:20000, :], Y[10000:20000]

depths = range(1,16,1)
leaves = np.power(2, range(2,13,1))

parameters = []
for depth in depths:
    for leaf in leaves:
        parameters.append((depth, leaf))


trainingErrors = []
validationErrors = []
for parameter in parameters:
    print "Training with maxDepth = ", parameter[0], " and minLeaves = ", parameter[1]
    dt = ml.dtree.treeClassify(Xtr, Ytr, maxDepth=parameter[0], minLeaf=parameter[1])
    trainingErrors.append(dt.err(Xtr, Ytr))
    validationErrors.append(dt.err(Xval, Yval))

index = np.argmin(validationErrors)
print "Minimum Error on Varying Parameters = ", validationErrors[index]
print "Parameters for Minimum Error = ", parameters[index]
