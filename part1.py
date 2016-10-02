import numpy as np
import matplotlib.pyplot as plt
iris = np.genfromtxt("data/iris.txt",delimiter=None)
Y = iris[:,-1] # target value is the last column
X = iris[:,0:-1] # features are the other columns

# (a) Displaying the number of features and data points
print 'Number of features\t=\t', X.shape[1]
print 'Number of data points\t=\t', X.shape[0]

# (b) Displaying the histogram for each feature, with 20 bins
bins = 20 # Change the number of bins in the histogram here
for i in range(0,X.shape[1]):
    plt.hist(X[:,i], bins)
    plt.show()

# (c) Computing the mean value of each feature
mean = np.mean(X, axis=0) # Computes the mean along every feature
print 'Mean value of features\t=\t', mean


# (d) Computing the variance and standard deviation of each feature
std = np.std(X, axis=0) # Computes the standard deviation along every feature
print 'Standard Deviation of features\t=\t', std
variance = np.var(X, axis = 0) # Computes the variance along every feature
print 'Variance of features\t=\t', variance

# (e) Normalizing the features by subtracting their mean and dividing by its
#       standard deviation

normFeatures = (X[:,:] - mean)/std
print 'Mean of normalized features\t=\t', np.mean(normFeatures, axis=0)
print 'Variance of normalized features\t=\t', np.var(normFeatures, axis=0)

# (f) Scatter plot for pair of features - (1,2) (1,3) (1,4)
colors = ['b','g','r']
for i in range(1,4):
    for c in np.unique(Y): # Iterates through every unique label
        plt.scatter( X[Y==c,0], X[Y==c,i], color=colors[int(c)])
    plt.show()
