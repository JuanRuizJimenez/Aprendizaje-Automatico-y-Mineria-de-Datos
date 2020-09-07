from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm as supportVectorMachine
from sklearn.model_selection import train_test_split

def pinta_puntos(X,y):
    pos = np.where(y == 1)
    plt.scatter(X[pos, 0], X[pos, 1], marker = '+', c = 'black')

    neg = np.where(y == 0)
    plt.scatter(X[neg, 0], X[neg, 1], marker = 'o', c = 'black', s = 30)
    plt.scatter(X[neg, 0], X[neg, 1], marker = 'o', c = 'yellow', s = 20)

def pinta_frontera_recta(X, Y, model):
    w = model.coef_[0]
    a = -w[0] / w[1]
   
    xx = np.array([X[:,0].min(), X[:,0].max()])
    yy = a * xx - (model.intercept_[0]) / w[1]
    
    plt.plot(xx, yy, color='blue')

def gaussianKernel(X1, X2, sigma):
    Gram = np.zeros((X1.shape[0], X2.shape[0]))
    for i, x1 in enumerate(X1):
        for j, x2 in enumerate(X2):
            x1 = x1.ravel()
            x2 = x2.ravel()
            Gram[i, j] = np.exp(-np.sum(np.square(x1 - x2)) / (2 * (sigma**2)))
    return Gram

def pinta_frontera_curva(X, y, model, sigma):
   
    x1plot = np.linspace(X[:,0].min(), X[:,0].max(), 100).T
    x2plot = np.linspace(X[:,1].min(), X[:,1].max(), 100).T
    X1, X2 = np.meshgrid(x1plot, x2plot)
    vals = np.zeros(X1.shape)
    for i in range(X1.shape[1]):
        this_X = np.column_stack((X1[:, i], X2[:, i]))
        vals[:, i] = model.predict(gaussianKernel(this_X, X, sigma))

    plt.contour(X1, X2, vals, colors="green", linewidths=0.3)

def test(svm, Xval, Yval):
    pred = svm.predict(Xval)
    acc = np.mean((pred == Yval).astype(int))

    return acc

def parte1():
    data = loadmat("proyecto_final_data_TRAIN.mat")

    X = data["X"]
    y = data["y"]
    Xval = data["Xval"]
    yval = data["yval"]
    yravel = y.ravel()

    Coef = 1.0

    svm = supportVectorMachine.SVC(kernel="linear", C=Coef)
    svm = svm.fit(X, yravel)

    prec = test(svm, Xval, Yval)

    print(prec)

def parte2():
    data = loadmat("proyecto_final_data_TRAIN.mat")
    
    X = data["X"]
    y = data["y"]
    Xval = data["Xval"]
    yval = data["yval"]
    yravel = y.ravel()
    
    Coef = 1.0
    sigma = 0.1

    svm = supportVectorMachine.SVC(C=Coef, kernel='precomputed', tol= 1e-3, max_iter= 100)
    svm = svm.fit(gaussianKernel(X, X, sigma=sigma), yravel)

    prec = test(svm, Xval, Yval)

    print(prec)

def main():

    #PARTE 1
    parte1()

    #PARTE 2
    #parte2()

main()