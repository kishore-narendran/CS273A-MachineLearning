import numpy as np
import matplotlib.pyplot as plt
import mltools as ml
import cPickle as pickle
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

pca_flag = True
quadraticFeature = True
nQuadraticFeatures = 3

X = np.genfromtxt("data/X_train.txt", delimiter = None)
Y = np.genfromtxt("data/Y_train.txt", delimiter = None)
X_test = np.genfromtxt("data/X_test.txt", delimiter = None)

X_all = np.concatenate((X,X_test), axis=0)
X_all, _ = ml.rescale(X_all)

print 'Performing PCA Analysis'

if pca_flag:
    pca = PCA(n_components=14)

    X_all_pca = pca.fit_transform(X_all)
    if quadraticFeature:
        for i in range(nQuadraticFeatures - 1, 0, -1):
            X_all_pca = np.column_stack(((X_all_pca[:,i] ** 2), X_all_pca))
    print X_all_pca.shape
    X_pca = X_all_pca[:200000,:]
    X_test = X_all_pca[200000:,:]

models = []
validationErrors = []
features = []
for i in range(1,X_pca.shape[1],1):
    print 'Logistic Regression with ', (i+1), ' features'
    if not pca_flag:
        Xtr, Ytr = X[:180000, :i], Y[:180000]
        Xval, Yval = X[180000:, :i], Y[180000:]
    else:
        Xtr, Ytr = X_pca[:180000,:i], Y[:180000]
        Xval, Yval = X_pca[180000:,:i], Y[180000:]
    model = LogisticRegression()
    model = model.fit(Xtr,Ytr)
    print(model.score(Xtr,Ytr))
    predict = model.predict(Xval)
    er = float(np.sum(predict!=Yval))/float(Yval.shape[0])
    print("Error rate for validation data set =",er)
    models.append(model)
    validationErrors.append(er)

# Finding the model with the least validation error
index = np.argmin(validationErrors)
model = models[index]

with open("loggress_model_quad.mdl", "wb") as output_file:
    pickle.dump(model, output_file)

# Making predictions with the model
X_test = X_test[:,0:index+1]
Ypred = model.predict_proba(X_test)
np.savetxt('Yhat_project_loggres.txt',
np.vstack( (np.arange(len(Ypred)) , Ypred[:,1]) ).T,
'%d, %.2f',header='ID,Prob1',comments='',delimiter=',')
