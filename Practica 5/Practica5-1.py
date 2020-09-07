import numpy as np
from scipy.io import loadmat
import scipy.optimize as opt
import matplotlib.pyplot as plt

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

def pinta_puntos(X, Y):
    plt.scatter(X, Y, marker = 'x', c = 'red', label = 'Entrada')

def pinta_Recta(T, x, y):
    x =  np.arange(np.min(x[:, 1]), np.max(x[:, 1]), 0.1)
    y = x.copy()

    a = len(x)
    i = 0

    while (i < a):
            y[i] = (x[i] * T[1]) + T[0]
            i = i + 1
    
    plt.plot(x, y, c='blue')

def main():
    data = loadmat("ex5data1.mat")

    X = data["X"]
    y = data["y"]

    landa = 0

    XwithOnes=np.hstack((np.ones(shape=(X.shape[0],1)),X))

    theta = np.ones(XwithOnes.shape[1])

    thetaOpt = calcOptTheta(XwithOnes, y, landa, theta)

    pinta_puntos(X, y)
    pinta_Recta(thetaOpt, XwithOnes, y)

    plt.legend()
    plt.show()

main()