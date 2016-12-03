import numpy as np
import matplotlib.pyplot as plt
import mltools as ml

iris = np.genfromtxt("data/iris.txt", delimiter = None)
X = iris[:,0:-1]

# (a) Loading the first two features of the iris data set and plotting to check
#   clustering
X_two = X[:,0:2]
plt.scatter(X[:,0], X[:,1], c='b')
plt.title('Plotting the first two features')
plt.show()


# (b) Running k means with k=5 and k=20 and plotting the same
k_clusters = [5, 20]
for k in k_clusters:
    (z, c, sumd) = ml.cluster.kmeans(X_two, k)
    ml.plotClassify2D(None, X_two, z)
    plt.title('k-Means Clustering with k = ' + str(k))
    plt.show()

initializations = ['random', 'farthest', 'k++']
parameters = []
for initialization in initializations:
    for k in k_clusters:
        parameters.append((k, initialization))


sumd_s = []
z_s = []
for parameter in parameters:
    (z, c, sumd) = ml.cluster.kmeans(X_two, parameter[0], parameter[1])
    z_s.append(z)
    sumd_s.append(sumd)

index = np.argmin(sumd_s)
ml.plotClassify2D(None, X_two, z_s[index])
plt.title("k = " + str(parameters[index][0]) + " initialization = " + parameters[index][1])
plt.show()




# (c) Running agglomerative clustering on the data with k = 5, and k = 20, with
#   with single linkage and complete linkage
linkage = {'min': 'Single Linkage', 'max': 'Complete Linkage'}
for k in k_clusters:
    for method in ['min', 'max']:
        (z, join) = ml.cluster.agglomerative(X_two, k, method=method)
        ml.plotClassify2D(None, X_two, z)
        plt.title('Agglomerative clustering with k = '+str(k) + ' and ' + linkage[method])
        plt.show()
