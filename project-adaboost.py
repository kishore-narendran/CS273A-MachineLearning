import numpy as np
import matplotlib.pyplot as plt
import mltools as ml
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import cPickle as pickle

X = np.genfromtxt("data/X_train.txt", delimiter = None)
Y = np.genfromtxt("data/Y_train.txt", delimiter = None)

X, Y = ml.shuffleData(X,Y)
def TrainOneDecisionTree():
    Xtr, Ytr = X[:10000, :], Y[:10000]
    Xval, Yval = X[10000:20000, :], Y[10000:20000]

    depths = range(2,22,1)
    features = range(1,15,1)
    parameters = []
    for depth in depths:
        for feature in features:
            parameters.append((depth, feature))
    validationErrors = []
    for parameter in parameters:
        decisionTree = DecisionTreeClassifier(max_depth=parameter[0], max_features=parameter[1])
        decisionTree.fit(Xtr, Ytr)
        predict = decisionTree.predict(Xval)
        er = float(np.sum(predict!=Yval))/float(Yval.shape[0])
        validationErrors.append(er)

    minIndex = np.argmin(validationErrors)
    print 'Best Validation Error = ', validationErrors[minIndex]
    print 'Parameters with Best Validation Performance = ', parameters[minIndex], minIndex

def TrainEnsemble():
    Xtr, Ytr = X[:10000, :], Y[:10000]
    Xval, Yval = X[10000:20000, :], Y[10000:20000]

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
                Xi, Yi = ml.bootstrapData(Xtr,Ytr, n_boot=10000)
                decisionTrees[i] = DecisionTreeClassifier(max_features=nFeature)
                decisionTrees[i] = decisionTrees[i].fit(Xi, Yi)
                # decisionTrees[i] = ml.dtree.treeClassify(Xi, Yi, maxDepth=16, minLeaf=256, nFeatures=nFeature)

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

Xtr, Ytr = X[:180000, :], Y[:180000]
Xval, Yval = X[180000:, :], Y[180000:]
#
# depths = range(2,22,1)
# features = range(1,15,1)
# estimators = [5, 10, 25, 50, 75]
# parameters = []
# for depth in depths:
#     for feature in features:
#         for estimator in estimators:
#             parameters.append((depth, feature, estimator))
#
# validationErrors = []
# for parameter in parameters:
#     model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=parameter[0], max_features=parameter[1]), n_estimators=parameter[2])
#     model = model.fit(Xtr, Ytr)
#     predict = model.predict(Xval)
#     er = float(np.sum(predict!=Yval))/float(Yval.shape[0])
#     validationErrors.append(er)
# minIndex = np.argmin(validationErrors)

model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=10, max_features=14), n_estimators=50)
model = model.fit(Xtr, Ytr)
predict = model.predict(Xval)
er = float(np.sum(predict!=Yval))/float(Yval.shape[0])
print 'Best Validation Error = ', er

with open("adaboost.mdl", "wb") as output_file:
    pickle.dump(model, output_file)
