import numpy as np
import matplotlib.pyplot as plt
import mltools as ml


# Problem 2 - Random Forests
X = np.genfromtxt("data/X_train.txt", delimiter = None)
Y = np.genfromtxt("data/Y_train.txt", delimiter = None)

Xtr, Ytr = X[:10000, :], Y[:10000]
Xval, Yval = X[10000:20000, :], Y[10000:20000]

bags = [1,5,10,25,45,60,75]
bagTrainError = []
bagValidationError = []
ensembles = []
for bag in bags:
    decisionTrees = [None]*bag
    trainingError = []
    for i in range(0,bag,1):
        # Drawing a random training sample every single time
        Xi, Yi = ml.bootstrapData(Xtr,Ytr, n_boot=10000)
        decisionTrees[i] = ml.dtree.treeClassify(Xi, Yi, maxDepth=16, minLeaf=4, nFeatures=6)

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

plt.plot(bags, bagTrainError, 'r', label='Training Error')
plt.plot(bags, bagValidationError, 'g', label='Validation Error')
plt.legend(loc='upper right')
plt.title('Error vs # of Learners in Bag')
plt.show()

index = np.argmin(bagValidationError)
print "Minimum Error on an Ensemble of Learners = ", validationErrors[index]
print "Number of Learners in Ensemble = ", bags[index]

ensemble = ensembles[index]
Xtest =  np.genfromtxt("data/X_test.txt", delimiter = None)
Ypred = np.zeros((Xtest.shape[0], 2))
for dt in ensemble:
    Ypred += dt.predictSoft(Xtest)

Ypred = Ypred/float(len(ensemble))
np.savetxt('Yhat_dtree_ensemble.txt',
np.vstack( (np.arange(len(Ypred)) , Ypred[:,1]) ).T,
'%d, %.2f',header='ID,Prob1',comments='',delimiter=',')
