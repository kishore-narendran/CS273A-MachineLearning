import numpy as np
import matplotlib.pyplot as plt
import mltools as ml

# (a) Loading the data from the curve80.txt file, and splitting to
#       75-25, training and test data
data = np.genfromtxt("data/curve80.txt",delimiter=None)
X = data[:, 0] # First column is feature
X = X[:,np.newaxis] # code expects shape (M,N) so make sure it's 2-dimensional
Y = data[:, 1] # Second column is the result
Xtr,Xte,Ytr,Yte = ml.splitData(X,Y,0.75) # split data set 75/25
Ytr = Ytr[:, np.newaxis]
Yte = Yte[:, np.newaxis]

# (b) Plotting the linear regression prediction function, and training data,
#       finding the regression coefficients, finding MSE of train and test data
lr = ml.linear.linearRegress(Xtr, Ytr) # create and train model
xs = np.linspace(0, 10, 200) # densely sample possible x-values
xs = xs[:, np.newaxis] # force "xs" to be an Mx1 matrix
ys = lr.predict(xs) # make predictions at xs
plt.scatter(Xtr, Ytr, c = 'red') # Plotting the training data points
plt.plot(xs, ys, c= 'black') # Plotting the predictor line
plt.title('Regression Function')
plt.show()
print 'Regression Coefficients\t=\t', lr.theta
YTrainPred = lr.predict(Xtr)
YTestPred = lr.predict(Xte)
mseTrain = np.mean((YTrainPred - Ytr) ** 2)
mseTest = np.mean((YTestPred - Yte) ** 2)
print 'Mean Square Error on Training Data\t=\t', mseTrain
print 'Mean Square Error on Test Data\t=\t', mseTest

# (c) Fitting y = f(x) with polynomial functions of increasing order
degrees = [1, 3, 5, 7, 10, 18]
trainingError = []
testError = []
for degree in degrees:
    # Scaling and making polynomial features
    XtrP = ml.transforms.fpoly(Xtr, degree, bias=False)
    XtrP, params = ml.transforms.rescale( XtrP )
    lr = ml.linear.linearRegress( XtrP, Ytr )
    XteP,_ = ml.transforms.rescale( ml.transforms.fpoly(Xte, degree, False ), params )
    xsP,_ = ml.transforms.rescale( ml.transforms.fpoly(xs, degree, False ), params )

    # Predicting for xs, and plotting predictor function
    ys = lr.predict(xsP) # make predictions at xs
    plt.scatter(Xtr, Ytr, c = 'red')
    ax = plt.axis()
    plt.plot(xs, ys, c = 'black') # Plotting the predictor line
    plt.axis(ax)
    plt.title("Polynomial Function of Degree " + str(degree))
    plt.show()

    # Calculating error in test and training data
    YTrainPredP = lr.predict(XtrP)
    YTestPredP = lr.predict(XteP)
    trainingError.append(np.mean((YTrainPredP - Ytr) ** 2))
    testError.append(np.mean((YTestPredP - Yte) ** 2))

plt.semilogy(degrees, trainingError, c = 'red')
plt.semilogy(degrees, testError, c = 'green')
plt.xticks(degrees, degrees)
plt.title("Training and Test Error vs Degree of Polynomial")
plt.show()
