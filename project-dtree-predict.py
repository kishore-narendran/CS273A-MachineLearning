import numpy as np
import matplotlib.pyplot as plt
import mltools as ml
import cPickle as pickle

def getRandomForestPred():

    with open("dtree.mdl", "rb") as input_file:
        e = pickle.load(input_file)

    ensemble = e[5]
    Xtest =  np.genfromtxt("data/X_test.txt", delimiter = None)
    Ypred = np.zeros((Xtest.shape[0], 2))
    for dt in ensemble:
        Ypred += dt.predictSoft(Xtest)

    Ypred = Ypred/float(len(ensemble))


Xtest =  np.genfromtxt("data/X_test.txt", delimiter = None)

with open("adaboost.mdl", "rb") as input_file:
    e = pickle.load(input_file)

Ypred = e.predict_proba(Xtest)
np.savetxt('Yhat_boost.txt',
np.vstack( (np.arange(len(Ypred)) , Ypred[:,1]) ).T,
'%d, %.2f',header='ID,Prob1',comments='',delimiter=',')
