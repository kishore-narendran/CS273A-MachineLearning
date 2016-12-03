import numpy as np
import matplotlib.pyplot as plt
import mltools as ml
from scipy import linalg
import random

X = np.genfromtxt("data/faces.txt", delimiter=None) # load face dataset
plt.figure()
img = np.reshape(X[0,:],(24,24)) # convert vectorized data point to 24x24 image patch
plt.imshow( img.T , cmap="gray")
plt.show()

# (a) Subtracting the mean of the face images from every face
X_mean = np.mean(X, axis=0)
X = X - X_mean # Converting the data to zero mean

# (b) Finding the SVD of the image features matrix
U, s, Vh = linalg.svd(X, full_matrices=False)
X0 = U.dot(np.diag(s)).dot(Vh)

# (c) Finding the approximation of X0 in eigen directions k = [1,10]
eigendirections = range(0,10,1)
error = []
for k in eigendirections:
    Xhat = U[:,0:k].dot(np.diag(s[0:k])).dot(Vh[0:k,:])
    error.append(np.mean((X0-Xhat) ** 2))

plt.plot(eigendirections, error, c='r')
plt.xlabel('K')
plt.ylabel('MSE')
plt.title('Error vs K')
plt.show()


# (d) Displaying the first three principal directions of the data
W = U.dot(np.diag(s)) # Calculating W as U.s for convenience
for j in range(0,3,1):
    alpha = 2*np.median(np.abs(W[:,j]))
    principal_image = X_mean + alpha*Vh[j,:]
    img = np.reshape(principal_image, (24, 24))
    plt.subplot(1,2,1)
    plt.imshow(img.T, cmap='gray')
    plt.title('Principal Direction #' + str(j) + '(+ alpha)')

    principal_image = X_mean - alpha*Vh[j,:]
    img = np.reshape(principal_image, (24, 24))
    plt.subplot(1,2,2)
    plt.imshow(img.T, cmap='gray')
    plt.title('Principal Direction #' + str(j) + '(-alpha)')
    plt.show()


# (e) Finding the first K = [5,10,50] eigen faces and using that to
#   reconstructions
images = [random.randrange(0, X.shape[0], 1) for x in range(0,2,1)]
for i in images:
    for k in [5, 10, 50]:
        img = W[i:i+1,0:k].dot(Vh[0:k,:])
        img = X_mean + img
        img = np.reshape(img, (24,24))
        plt.subplot(1,2,1)
        plt.title('Reconstructed Image i = ' + str(i) + ' with k = ' + str(k))
        plt.imshow(img.T, cmap='gray')

        plt.subplot(1,2,2)
        img = np.reshape(X[i,:],(24,24))
        plt.title('Original Image i = ' + str(i))
        plt.imshow(img.T, cmap='gray')
        plt.show()

# (f) Latent Space Models
idx = [random.randrange(0, X.shape[0], 1) for x in range(0,25,1)]
coord,params = ml.transforms.rescale( W[:,0:2] ) # normalize scale of "W" locations
for i in idx:
    loc = (coord[i,0],coord[i,0]+0.5, coord[i,1],coord[i,1]+0.5) # where to place the image & size
    img = np.reshape( X[i,:], (24,24) )
    plt.imshow( img.T , cmap="gray", extent=loc ) # draw each image
    plt.axis((-2,2,-2,2))
plt.show()
