import numpy as np
from scipy.io import loadmat
import scipy.optimize as opt
import matplotlib.pyplot as plt

def sigmoide(x):
    return 1 / (1 + np.exp(-x))

#Función para el coste
def coste(theta, X, y, landa, m):
    h = np.dot(X, theta[:, None])

    thetaAux = np.delete(theta, 0, 0)
    return ((1 / (2 * m)) * (np.sum(np.square(h - y)))) + ((landa / (2 * m)) * np.sum(np.square(thetaAux)))
    
#Función para calculo de gradiente
def gradiente(theta, XX, Y, landa, m):
    h = np.dot(XX, theta[:, None])

    #Eliminamos la primera columna de theta
    thetaAux = np.delete(theta, 0, 0)
    return (1 / m) * np.matmul(XX.T, h - Y)+((landa/m) * thetaAux)

def calcOptTheta(X, Y, landa, theta):
    result = opt.fmin_tnc(func=coste, x0=theta, fprime=gradiente, args=(X, Y, landa, X.shape[0]))
    return result[0]

def curva_aprendizaje(X, y, landa, theta, Xval, yval):

    err1 = np.zeros((len(X)))
    err2 = np.zeros((len(Xval)))

    i = 1
    while (i < len(X) + 1):
        thetas = calcOptTheta(X[0:i], y[0:i], landa, theta)
        
        err1[i - 1] = calc_error(thetas, X[0:i], y[0:i], landa, len(X))
        err2[i - 1] = calc_error(thetas, Xval, yval, landa, len(Xval))
        i += 1   

    return err1, err2    

def calc_error(theta, X, y, landa, m):
    h = np.dot(X, theta[:, None])
    return (1 / (2 * m)) * ((np.sum(np.square(h - y))))

def pinta_Curva_Aprendizaje(err1, err2):
    
    a = np.arange(len(err1))
    b = err1
    plt.plot(a, b, c="blue", label="Train")

    d = err2[0:len(err1)]
    plt.plot(a, d, c="orange", label="Cross Validation")

def main():
    data = loadmat("ex5data1.mat")

    X = data["X"]
    y = data["y"]
    Xval = data["Xval"]
    yval = data["yval"]

    landa = 0

    XwithOnes=np.hstack((np.ones(shape=(X.shape[0],1)),X))

    XvalWithOnes = np.hstack((np.ones(shape=(Xval.shape[0],1)),Xval))

    theta = np.ones(XwithOnes.shape[1])

    err1, err2 = curva_aprendizaje(XwithOnes, y, landa, theta, XvalWithOnes, yval)

    pinta_Curva_Aprendizaje(err1, err2)

    plt.legend()
    plt.show()

main()