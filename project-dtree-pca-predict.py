import numpy as np
import matplotlib.pyplot as plt
import mltools as ml
import cPickle as pickle
from sklearn.decomposition import PCA

X = np.genfromtxt("data/X_train.txt", delimiter = None)
Y = np.genfromtxt("data/Y_train.txt", delimiter = None)
X_test = np.genfromtxt("data/X_test.txt", delimiter = None)

X_all = np.concatenate((X,X_test), axis=0)
X_all, _ = ml.rescale(X_all)

n_components = 8
bag = 60

pca = PCA(n_components=n_components)

X_all_pca = pca.fit_transform(X_all)
X_pca = X_all_pca[:200000,:]
Xtest = X_all_pca[200000:,:]
Xtr, Ytr = X_pca[:180000,:], Y[:180000]
Xval, Yval = X_pca[180000:,:], Y[180000:]

print 'Training ', bag, ' decision tree(s)'

decisionTrees = [None]*bag
trainingError = []
for i in range(0,bag,1):
    # Drawing a random training sample every single time
    Xi, Yi = ml.bootstrapData(Xtr,Ytr, n_boot=180000)
    decisionTrees[i] = ml.dtree.treeClassify(Xi, Yi, maxDepth=16, minLeaf=256)

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

validationError = np.mean(YHatValidation != Yval)
trainError = np.mean(YHatTraining != Ytr)

print 'Validation Error = ', validationError
print 'Training Error = ', trainError

with open("dtree_pca_best.mdl", "wb") as output_file:
    pickle.dump(decisionTrees, output_file)

Ypred = np.zeros((Xtest.shape[0], 2))
for dt in decisionTrees:
    Ypred += dt.predictSoft(Xtest)

Ypred = Ypred/float(len(decisionTrees))
np.savetxt('Yhat_project_pca_ensemble.txt',
np.vstack( (np.arange(len(Ypred)) , Ypred[:,1]) ).T,
'%d, %.2f',header='ID,Prob1',comments='',delimiter=',')
