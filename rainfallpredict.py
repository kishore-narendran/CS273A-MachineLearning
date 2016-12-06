import numpy as np
#import matplotlib.pyplot as plt
import mltools as ml
#import mltools.dtree as dtree
from os import listdir
import pickle as p
import os
import time
import sys

def computeError(a, b):
    #print 'Sizes of matrices being compared: ', a.shape, b.shape
    #if a.shape[0] != b.shape[0] or a.shape[1] != b.shape[1]:
     #   raise Exception("Incorrect sizes!")
    return np.count_nonzero(a - b) / (1.0 * a.shape[0])

def init():
    Xdata = np.genfromtxt("data/X_train.txt")
    Ydata = np.genfromtxt("data/Y_train.txt")
    Xtedata = np.genfromtxt("data/X_test.txt")
    Ydata = Ydata[:,np.newaxis]
    return Xdata, Ydata, Xtedata

def exportData(fileName, Ypred):
    np.savetxt(fileName,
               np.vstack((np.arange(len(Ypred)), Ypred)).T,
               '%d, %.2f', header='ID,Prob1', comments='', delimiter=',');

def reload_from_disk(dir="default-models"):
    models = []
    if os.path.isdir(dir):
        for m in listdir(dir):
            if m.endswith(".mdl"):
                f = open(dir + "/" + m, 'r')
                models.append(p.load(f))
                f.close()
    return models

def save_models(models, dir="default-models", clear=False):
    if not os.path.isdir(dir):
        os.makedirs(dir)
    n = len(listdir(dir))
    if clear:
        for m in listdir(dir):
            os.remove(m)
        n = 0
    for i, m in enumerate(models):
        f = open(dir + "/" + str((i + n)) + '.mdl', 'w')
        p.dump(m, f)
        f.close()


def train_fold(X, Y, nFolds, nFeatures, depth, minLeaf):
    m = []
    errTr = []
    errTe = []
    print "NFOLD: " , X.shape, Y.shape
    for iFold in range(nFolds):
        Xtri, Xtei, Ytri, Ytei = ml.crossValidate(X, Y, nFolds, iFold)
        print Xtri.shape, Ytri.shape, Xtei.shape, Ytei.shape
        dt = ml.dtree.treeClassify(Xtri, Ytri, minLeaf=minLeaf, maxDepth=depth, nFeatures=nFeatures)
        #m.append(dt)

        Yteihat = dt.predict(Xtei)
        Ytrihat = dt.predict(Xtri)

        errTr.append(computeError(Ytri, Ytrihat))
        errTe.append(computeError(Ytei, Yteihat))

        if (errTr[-1] > 0.32 and errTe[-1] > 0.32):
            print "High Err: ", (nFeatures, depth, minLeaf), "has high error. Stopping nFold training on these parameters"
            break

    return (np.mean(errTr), np.mean(errTe), m)

def select_train():
    Xdata, Ydata, Xtedata = init()
    models = []

    for leaf in [5, 7, 11, 13, 16]:
        start = time.time()
        dt = ml.dtree.treeClassify(Xdata, Ydata, maxDepth=17, minLeaf=leaf)
        end = time.time()

        print 'for leaf', leaf, 'time:', (end - start), 'seconds'
        models.append(dt)

    save_models(models, "selected-models")

def train(models, lower, upper, destination_folder):
    Xdata, Ydata, Xtedata = init()
    #X, Y = Xdata[0:10], Ydata[0:10]
    X, Y = Xdata, Ydata
    X, _ = ml.transforms.rescale(X)
    nFolds = 5
    trError = []
    testError = []
    thresholdError = 0.7
    nFolds = 5
    #leaves = [5, 7, 10, 13, 15, 18, 21, 24, 27, 30, 33, 36, 40]
    for nFeatures in range(lower, upper):
        for depth in [10, 15, 16, 17, 19, 21, 30, 45, 50]:
            for minLeaf in [5, 7, 10, 13, 20, 30, 64, 128, 150, 200, 250, 500, 1000, 1250]:
                #print 'depth', depth
                print 'Features, Depth, minLeaf, modelIndex: ', (nFeatures, depth, minLeaf, len(models))

                start = time.time()
                Xi, Yi = ml.bootstrapData(X, Y, X.shape[0])
                errTr, errTe, m = train_fold(Xi, Yi, nFolds, nFeatures, depth, minLeaf)
                end = time.time()
                #models.extend(m)

                trError.append(errTr)
                testError.append(errTe)

                print 'Average training erorr', trError[-1]
                print 'Average test error', testError[-1]
                print 'Total time for model', (end - start), 'Time per split: ', ((end - start) / (1.0 * nFolds))

                if testError[-1] < 0.29 or trError[-1] < 0.20:
                    print ':::LOW ERR::: (f,d,ml,len_model,teE, trE', \
                        (nFeatures, depth, minLeaf, len(models), testError[-1], trError[-1])

        # TODO: If if erorr is less than threshold, then add it to the models array

    # plt.plot(range(0, len(trError)), trError, 'b-')
    # plt.plot(range(0, len(testError)), testError, 'g-')
    # plt.show()

    f = open('training_error' + str(lower) + '_' +str(upper), 'w')
    p.dump(trError, f)
    f.close()
    f = open('test_error' + str(lower) + '_' +str(upper), 'w')
    p.dump(testError, f)
    f.close()

    save_models(models, destination_folder)

def train_from_triples(models, triple_file_name, destination_folder):
    Xdata, Ydata, Xtedata = init()
    Xs, _ = ml.transforms.rescale(Xdata);
    Ys = Ydata;
    Xtes, _ = ml.transforms.rescale(Xtedata);
    print '----Training models------'
    #Xi, Yi = Xdata[0:10000], Ydata[0:10000]
    #Xs, Ys = Xs[0:10000], Ys[0:10000]
    Xi, Yi = Xdata, Ydata
    Xs, Ys = Xs, Ydata
    f = open(triple_file_name, 'r')
    triples = f.readlines()
    f.close()

    for triple in triples:
        nf = int(triple.split(',')[0].strip())
        d = int(triple.split(',')[1].strip())
        l = int(triple.split(',')[2].strip())
        print 'Now Training (nf,d,ml):', nf,d,l
        #dt = ml.dtree.treeClassify(Xi, Yi, maxDepth=d, nFeatures=nf, minLeaf=l)
        #models.append(dt)
        #Ypred = dt.predict(Xi)
        #print 'Training error with triple on unscaled: ', triple.strip(), 'is', computeError(Ypred[:,np.newaxis], Yi)
        Xi, Yi = ml.bootstrapData(Xs, Ys, Xs.shape[0])
        dt = ml.dtree.treeClassify(Xi, Yi, maxDepth=d, nFeatures=nf, minLeaf=l)
        Ypred = dt.predict(Xs)
        print 'Training error with triple on scaled: ', triple.strip(), 'is', computeError(Ypred[:,np.newaxis], Ys)
        models.append(dt)

    #save_models(models, destination_folder)
    print '-----Predicting the scores------'
    kaggle_predict(models, True)

def kaggle_predict(models, rescale=False):
    print 'Finished loading', len(models), "models"
    print 'Rescale Enabled?', rescale
    Xdata, Ydata, Xtedata = init()
    if (rescale):
        Xtedata, _ = ml.transforms.rescale(Xtedata)
    predictTe = np.zeros((Xtedata.shape[0], len(models)))
    for i, model in enumerate(models):
        print 'Predicting with model #' + str(i)
        predictTe[:, i] = model.predictSoft(Xtedata)[:,1]
    Ytepred = np.mean(predictTe, axis=1)
    exportData('kaggle_dec2.txt', Ytepred)

def load_all_triple_models():
    m = []
    m.extend(reload_from_disk("triple-1"))
    m.extend(reload_from_disk("triple-2"))
    m.extend(reload_from_disk("triple-3"))
    m.extend(reload_from_disk("triple-4"))
    m.extend(reload_from_disk("selected-models"))
    kaggle_predict(m)

def testModels(models):
    Xdata, Ydata, Xtedata = init()
    for i, m in enumerate(models):
        Ydatahat = m.predict(Xdata)
        print 'error on model', i, computeError(Ydatahat[:,np.newaxis], Ydata)


def ada_boost(models, X, Y):
    wts = []
    predictTe = np.zeroes((X.shape[0], len(models)))
    for i, model in enumerate(models):
        pass


if __name__ == '__main__':
    #select_train()
    train([], int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
    #train([], 1, 5, 'test-mdoels2')
    #kaggle_predict(reload_from_disk("selected-models"))
    #train_from_triples(reload_from_disk("triple_selected"), 'triples', 'triples-1')
    #load_all_triple_models()
    #train_from_triples([], 'triple-new', 'scaled-triples-new')
    #kaggle_predict(reload_from_disk("scaled-triples-new"), True)