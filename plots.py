from itertools import combinations
import rainfallpredict as r
import mltools as ml
import matplotlib.pyplot as plt

def scatter():
    X, Y, Xte = r.init()
    #for x in combinations(range(14), 2):
    #    #plt.subplot(1, 14, x[1])
    #    plt.scatter(X[:,x[0]], X[:,x[1]], c=Y)
    #    plt.show()

    # everything wrt feature 1
    const_feature = 0
    Xtri, Xtei, Ytri, Ytei = ml.crossValidate(X, Y, 5, 0)
    for i in range(1, 14):
        if i != const_feature:
            plt.scatter(Xtei[:,const_feature], Xtei[:,i], c=Ytei)
            plt.show()

def hists():
    X, Y, Xte = r.init()
    Xtri, Xtei, Ytri, Ytei = ml.crossValidate(X, Y, 5, 0)
    # for i in range(0, 4):
    #     plt.subplot(1, 4, i+1)
    #     plt.hist(Xtei[:,i])
    # plt.show()
    for i in range(Xtei.shape[0]):
        plt.hist(Xtei[:,i])
        plt.show()

if __name__ == '__main__':
    hists()