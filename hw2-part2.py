import numpy as np
import matplotlib.pyplot as plt
import mltools as ml

data = np.genfromtxt("data/curve80.txt",delimiter=None)
X = data[:, 0] # First column is feature
X = X[:,np.newaxis] # code expects shape (M,N) so make sure it's 2-dimensional
Y = data[:, 1] # Second column is the result
Xtr,Xte,Ytr,Yte = ml.splitData(X,Y,0.75) # split data set 75/25

nFolds = 5;
degrees = [1, 3, 5, 7, 10, 18]
validationMSEs = []
for degree in degrees:
    J = []
    for iFold in range(nFolds):
        # ith block as validation
        Xti,Xvi,Yti,Yvi = ml.crossValidate(Xtr,Ytr,nFolds,iFold)
        Yvi = Yvi[:, np.newaxis]
        XtiP = ml.transforms.fpoly(Xti, degree, bias=False)
        XtiP, params = ml.transforms.rescale( XtiP )
        learner = ml.linear.linearRegress( XtiP, Yti )
        XviP,_ = ml.transforms.rescale( ml.transforms.fpoly(Xvi, degree, False ), params )

        # Calculating error in test and training data
        YValPredP = learner.predict(XviP)
        valError = np.mean((YValPredP - Yvi) ** 2)
        J.append(valError)
    validationMSEs.append(np.mean(J))
plt.semilogy(degrees, validationMSEs, c = 'red')
plt.xticks(degrees, degrees)
plt.show()
