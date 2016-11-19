import numpy as np
import matplotlib.pyplot as plt
import mltools as ml


# Problem 2 - Decision Trees on Kaggle
# (a) - Load the training data, and split into the training and validation
#   data: first 10000 for training and the next 10000 for validation
X = np.genfromtxt("data/X_train.txt", delimiter = None)
Y = np.genfromtxt("data/Y_train.txt", delimiter = None)

Xtr, Ytr = X[:10000, :], Y[:10000]
Xval, Yval = X[10000:20000, :], Y[10000:20000]

# (b) Learn a decision tree classifier on the data, with a maxDepth of 50
print 'Training a decision tree with depth = 50'
dt = ml.dtree.treeClassify(Xtr, Ytr, maxDepth=50)
trainingError = dt.err(Xtr, Ytr)
validationError = dt.err(Xval, Yval)
print 'Training Error:', trainingError
print 'Validation Error:', validationError
print '='*75

# (c) Vary the depth of the decision tree and note the training and Validation
#   errors on the same
maxDepths = range(1,16,1)
trainingErrors = []
validationErrors = []
for depth in maxDepths:
    # print 'Training a decision tree of depth = ', depth
    dt = ml.dtree.treeClassify(Xtr, Ytr, maxDepth=depth)
    trainingErrors.append(dt.err(Xtr, Ytr))
    validationErrors.append(dt.err(Xval, Yval))

plt.plot(maxDepths, trainingErrors, 'r', label='Training Error')
plt.plot(maxDepths, validationErrors, 'g', label='Validation Error')
plt.xlabel('Max Depth')
plt.ylabel('Error Rate')
plt.title('Error Rate vs Depth')
plt.show()

index = np.argmin(validationErrors)
print "Minimum Error on Varying Depth = ", validationErrors[index]
print "Depth for Minimum Error = ", maxDepths[index]
print '='*75

# (d) Vary the minimum leaves parameter decision tree and note the training
#   and Validation errors on the same
minLeaves = range(2,13,1)
minLeaves = np.power(2,minLeaves)
trainingErrors = []
validationErrors = []
for leaves in minLeaves:
    # print 'Training a decision tree with minLeaves = ', leaves
    dt = ml.dtree.treeClassify(Xtr, Ytr, maxDepth=50, minLeaf=leaves)
    trainingErrors.append(dt.err(Xtr, Ytr))
    validationErrors.append(dt.err(Xval, Yval))

plt.plot(minLeaves, trainingErrors, 'r', label='Training Error')
plt.plot(minLeaves, validationErrors, 'g', label='Validation Error')
plt.xlabel('Min Leaves')
plt.ylabel('Error Rate')
plt.title('Error Rate vs # of Leaves')
plt.show()

index = np.argmin(validationErrors)
print "Minimum Error on Varying MinLeaves = ", validationErrors[index]
print "MinLeaves for Minimum Error = ", minLeaves[index]
print '='*75

# (e) Compute and plot the ROC curve for the trained model and the AUC score
#   for the trained model

dt = ml.dtree.treeClassify(Xtr, Ytr, maxDepth=7, minLeaf=4)

errTrain = dt.err(Xtr, Ytr)
errVal = dt.err(Xval, Yval)

print 'Training Error:', errTrain
print 'Validation Error:', errVal

print 'AUC Training Data:', dt.auc(Xtr, Ytr)
print 'AUC Validation Data:', dt.auc(Xval, Yval)

fprTrain, tprTrain, tnrTrain = dt.roc(Xtr, Ytr)
fprValidation, tprValidation, tnrValidation = dt.roc(Xval, Yval)

plt.plot(fprTrain, tprTrain, 'r')
plt.plot(fprValidation, tprValidation, 'g')
plt.title('ROC for Training and Validation Data')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
print '='*75

# (f) Using the best complexity model training with the first 20000 entries,
#   and predicting for all the entries in the X_test.txt data file
print "Training with 20000 entries and making predictions"
newXtr, newYtr = X[:20000, :], Y[:20000]
dt = ml.dtree.treeClassify(newXtr, newYtr, maxDepth=7, minLeaf=4)
Xte = np.genfromtxt("data/X_test.txt", delimiter = None)
Ypred = dt.predictSoft( Xte )
np.savetxt('Yhat_dtree_d5_l8_f6.txt',
np.vstack( (np.arange(len(Ypred)) , Ypred[:,1]) ).T,
'%d, %.2f',header='ID,Prob1',comments='',delimiter=',');
