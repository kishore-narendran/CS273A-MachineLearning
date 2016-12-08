import numpy as np
import matplotlib.pyplot as plt
import mltools as ml
import cPickle as pickle


# Problem 2 - Random Forests
X = np.genfromtxt("data/X_train.txt", delimiter = None)
Y = np.genfromtxt("data/Y_train.txt", delimiter = None)
X, Y = ml.shuffleData(X, Y)
X, _ = ml.rescale(X)

Xtr, Ytr = X[:180000, :], Y[:180000]
Xval, Yval = X[180000:, :], Y[180000:]
nFeatures = [5,6,7,8,9,10,11,12,13,14]
for nFeature in nFeatures:
    print "=" * 50
    print 'Training Decision Trees with ', str(nFeature), ' features'
    bags = [1,5,10,25,45,60,75]
    bagTrainError = []
    bagValidationError = []
    ensembles = []
    for bag in bags:
        print 'Training ', bag, ' decision tree(s)'
        decisionTrees = [None]*bag
        trainingError = []
        for i in range(0,bag,1):
            # Drawing a random training sample every single time
            Xi, Yi = ml.bootstrapData(Xtr,Ytr, n_boot=180000)
            decisionTrees[i] = ml.dtree.treeClassify(Xi, Yi, maxDepth=16, minLeaf=256, nFeatures=nFeature)

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

    # plt.plot(bags, bagTrainError, 'r', label='Training Error')
    # plt.plot(bags, bagValidationError, 'g', label='Validation Error')
    # plt.legend(loc='upper right')
    # plt.title('Error vs # of Learners in Bag')
    # plt.show()

    with open("dtree_f" + str(nFeature) + ".mdl", "wb") as output_file:
        pickle.dump(ensembles, output_file)

    print "=" * 50


'''
==================================================
Training Decision Trees with  4  features
Training  1  decision tree(s)
Training  5  decision tree(s)
Training  10  decision tree(s)
Training  25  decision tree(s)
Training  45  decision tree(s)
Training  60  decision tree(s)
Training  75  decision tree(s)
Minimum Validation Error =  0.2923
Number of learners in Bag =  25
==================================================
==================================================
Training Decision Trees with  5  features
Training  1  decision tree(s)
Training  5  decision tree(s)
Training  10  decision tree(s)
Training  25  decision tree(s)
Training  45  decision tree(s)
Training  60  decision tree(s)
Training  75  decision tree(s)
Minimum Validation Error =  0.2908
Number of learners in Bag =  45
==================================================
==================================================
Training Decision Trees with  6  features
Training  1  decision tree(s)
Training  5  decision tree(s)
Training  10  decision tree(s)
Training  25  decision tree(s)
Training  45  decision tree(s)
Training  60  decision tree(s)
Training  75  decision tree(s)
Minimum Validation Error =  0.2896
Number of learners in Bag =  25
==================================================
==================================================
Training Decision Trees with  7  features
Training  1  decision tree(s)
Training  5  decision tree(s)
Training  10  decision tree(s)
Training  25  decision tree(s)
Training  45  decision tree(s)
Training  60  decision tree(s)
Training  75  decision tree(s)
Minimum Validation Error =  0.29025
Number of learners in Bag =  60
==================================================
==================================================
Training Decision Trees with  8  features
Training  1  decision tree(s)
Training  5  decision tree(s)
Training  10  decision tree(s)
Training  25  decision tree(s)
Training  45  decision tree(s)
Training  60  decision tree(s)
Training  75  decision tree(s)
Minimum Validation Error =  0.29
Number of learners in Bag =  45
==================================================
==================================================
Training Decision Trees with  9  features
Training  1  decision tree(s)
Training  5  decision tree(s)
Training  10  decision tree(s)
Training  25  decision tree(s)
Training  45  decision tree(s)
Training  60  decision tree(s)
Training  75  decision tree(s)
Minimum Validation Error =  0.28925
Number of learners in Bag =  45
==================================================
==================================================
Training Decision Trees with  10  features
Training  1  decision tree(s)
Training  5  decision tree(s)
Training  10  decision tree(s)
Training  25  decision tree(s)
Training  45  decision tree(s)
Training  60  decision tree(s)
Training  75  decision tree(s)
Minimum Validation Error =  0.28965
Number of learners in Bag =  45
==================================================
==================================================
Training Decision Trees with  11  features
Training  1  decision tree(s)
Training  5  decision tree(s)
Training  10  decision tree(s)
Training  25  decision tree(s)
Training  45  decision tree(s)
Training  60  decision tree(s)
Training  75  decision tree(s)
Minimum Validation Error =  0.28865
Number of learners in Bag =  75
==================================================
==================================================
Training Decision Trees with  12  features
Training  1  decision tree(s)
Training  5  decision tree(s)
Training  10  decision tree(s)
Training  25  decision tree(s)
Training  45  decision tree(s)
Training  60  decision tree(s)
Training  75  decision tree(s)
Minimum Validation Error =  0.2894
Number of learners in Bag =  25
==================================================
==================================================
Training Decision Trees with  13  features
Training  1  decision tree(s)
Training  5  decision tree(s)
Training  10  decision tree(s)
Training  25  decision tree(s)
Training  45  decision tree(s)
Training  60  decision tree(s)
Training  75  decision tree(s)
Minimum Validation Error =  0.28825
Number of learners in Bag =  75
==================================================
==================================================
Training Decision Trees with  14  features
Training  1  decision tree(s)
Training  5  decision tree(s)
Training  10  decision tree(s)
Training  25  decision tree(s)
Training  45  decision tree(s)
Training  60  decision tree(s)
Training  75  decision tree(s)
Minimum Validation Error =  0.2885
Number of learners in Bag =  75
==================================================
'''
