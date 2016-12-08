import numpy as np
import matplotlib.pyplot as plt
import mltools as ml
import cPickle as pickle


# Problem 2 - Random Forests
X = np.genfromtxt("data/X_train.txt", delimiter = None)
Y = np.genfromtxt("data/Y_train.txt", delimiter = None)

Xtr, Ytr = X[:180000, :], Y[:180000]
Xval, Yval = X[180000:, :], Y[180000:]

bags = [1,5,10,25,45,60,75]
bagTrainError = []
bagValidationError = []
ensembles = []
for bag in bags:
    print 'Training ', bag, ' decision trees'
    decisionTrees = [None]*bag
    trainingError = []
    for i in range(0,bag,1):
        # Drawing a random training sample every single time
        Xi, Yi = ml.bootstrapData(Xtr,Ytr, n_boot=180000)
        decisionTrees[i] = ml.dtree.treeClassify(Xi, Yi, maxDepth=16, minLeaf=256, nFeatures=9)

    YHatValidation = np.zeros((Xval.shape[0], bag))
    YHatTraining = np.zeros((Xtr.shape[0], bag))
    for i in range(0,len(decisionTrees),1):
        decisionTree = decisionTrees[i]
        YHatValidation[:, i] = decisionTree.predict(Xval)
        YHatTraining[:,i] = decisionTree.predict(Xtr)

    # YHatValidation = np.sum(YHatValidation, axis=1)/float(bag)
    YHatValidation = np.mean(YHatValidation, axis=1)
    YHatValidation[YHatValidation > 0.5] = 1
    YHatValidation[YHatValidation <= 0.5] = 0

    # YHatTraining = np.sum(YHatTraining, axis=1)/float(bag)
    YHatTraining = np.mean(YHatTraining, axis=1)
    YHatTraining[YHatTraining > 0.5] = 1
    YHatTraining[YHatTraining <= 0.5] = 0

    bagValidationError.append(np.mean(YHatValidation != Yval))
    bagTrainError.append(np.mean(YHatTraining != Ytr))

    ensembles.append(decisionTrees)

index = np.argmin(bagValidationError)
print 'Minimum Validation Error = ', bagValidationError[index]
print 'Number of learners in Bag = ', bags[index]

plt.plot(bags, bagTrainError, 'r', label='Training Error')
plt.plot(bags, bagValidationError, 'g', label='Validation Error')
plt.legend(loc='upper right')
plt.title('Error vs # of Learners in Bag')
plt.show()

with open("dtree.mdl", "wb") as output_file:
    pickle.dump(ensembles, output_file)

'''
Training  1  decision trees
Training  5  decision trees
Training  10  decision trees
Training  25  decision trees
Training  45  decision trees
Training  60  decision trees
Training  75  decision trees
'''
