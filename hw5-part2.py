import numpy as np
import matplotlib.pyplot as plt
import mltools as ml
from scipy import linalg

X = np.genfromtxt("data/faces.txt", delimiter=None) # load face dataset
# plt.figure()
# img = np.reshape(X[0,:],(24,24)) # convert vectorized data point to 24x24 image patch
# plt.imshow( img.T , cmap="gray")
# plt.show()

# (a) Subtracting the mean of the face images from every face
X_mean = np.mean(X, axis=0)
X = X - X_mean # Converting the data to zero mean

# (b) Finding the SVD of the image features matrix
U, s, Vh = linalg.svd(X, full_matrices=False)
X0 = U.dot(np.diag(s)).dot(Vh)

# (c) Finding the approximation of X0 in eigen directions k = [1,10]
eigendirections = range(1,11,1)
error = []
for k in eigendirections:
    Xhat = U[:,0:k].dot(np.diag(s[0:k])).dot(Vh[0:k,:])
    error.append(np.mean((X0-Xhat) ** 2))

plt.plot(eigendirections, error, c='r')
plt.title('Error vs K')
plt.show()


# (d) Displaying the first three principal directions of the data
W = U.dot(np.diag(s))
for j in range(0,3,1):
    alpha = 2*np.median(np.abs(W[:,j]))
    principal_image = X_mean + alpha*Vh[j,:]
    img = np.reshape(principal_image, (24, 24))
    plt.subplot(3,2,j*2 + 1)
    plt.imshow(img.T, cmap='gray')
    plt.title('Principal Image #' + str(j) + ' Adding')

    principal_image = X_mean - alpha*Vh[j,:]
    img = np.reshape(principal_image, (24, 24))
    plt.subplot(3,2,j*2 + 2)
    plt.imshow(img.T, cmap='gray')
    plt.title('Principal Image #' + str(j) + ' Subtracting')
plt.show()
