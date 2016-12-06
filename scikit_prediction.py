import numpy as np
import rainfallpredict as r
import mltools as ml
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier



def predict():
    rng = np.random.RandomState(1)
    X, Y, Xte = r.init()

    regr_1 = DecisionTreeClassifier(max_depth=16)
    regr_2 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=16, max_leaf_nodes=1024), n_estimators=100, random_state=rng)

    regr_1.fit(X, Y)
    regr_2.fit(X, Y)
    y1 = regr_1.predict(X)
    y2 = regr_2.predict(X)
    y1  = y1[:,np.newaxis]
    y2 = y2[:,np.newaxis]
    print 'Error with regr1', r.computeError(Y, y1)
    print 'Error with regr2', r.computeError(Y, y2)

    r.exportData('dec5-4.txt', regr_2.predict_proba(X)[:,1])

    return regr_1, regr_2

# chosen maxDepth = 16
# chosen maxLeafNodes = 1000
def predict2():
    X, Y, Xte = r.init();
    X, _ = ml.transforms.rescale(X)
    nFolds = 5;
    errTr = []
    errTe = []
    l = [2, 4, 6, 8, 10, 16, 32, 64, 100, 128, 150, 256, 328, 400, 512, 768, 800, 1024, 1568, 2048]
    for features in l:
        dtc = DecisionTreeClassifier(max_leaf_nodes=features)
        tre = 0
        tee = 0
        for iFold in range(nFolds):
            print 'Training for features', features, 'fold', iFold
            Xtri, Xtei, Ytri, Ytei = ml.crossValidate(X, Y, nFolds, iFold)
            Ytri, Ytei  = Ytri[:,np.newaxis], Ytei[:,np.newaxis]
            #print Xtri.shape, Xtei.shape, Ytri.shape, Ytei.shape
            dtc.fit(Xtri, Ytri)
            e1 = r.computeError(dtc.predict(Xtri)[:,np.newaxis], Ytri)
            tre += e1
            print 'Training Error', e1
            e1 = r.computeError(dtc.predict((Xtei))[:, np.newaxis], Ytei)
            print 'Test Error', e1
            tee += e1
        errTr.append(tre / nFolds);
        errTe.append(tee / nFolds);

        print '===== features:', features, 'Training: ', errTr[-1], 'Test', errTe[-1]
    print 'Training Error', errTr
    print 'Test Error', errTe

    plt.plot(l, errTr, 'r.')
    plt.plot(l, errTe, 'b.')
    plt.show()


if __name__ == '__main__':
    predict()
