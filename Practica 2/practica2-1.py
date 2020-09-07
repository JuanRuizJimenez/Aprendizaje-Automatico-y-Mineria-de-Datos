import numpy as np
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
import scipy.optimize as opt

def carga_csv(file_name):
    """carga el fichero csv especificado y lo
    devuelve en un array de numpy"""
    valores = read_csv(file_name, header=None).values
    # suponemos que siempre trabajaremos con float
    return valores.astype(float)

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def cost(theta, X, Y):
    H = sigmoid(np.matmul(X, theta))
    cost = (- 1 / (len(X))) * (np.dot(Y, np.log(H)) + np.dot((1 - Y), np.log(1 - H)))
    return cost

def gradient(theta, XX, Y):
    H = sigmoid(np.matmul(XX, theta))
    grad = (1 / len(Y)) * np.matmul(XX.T, H - Y)
    return grad
 
def pinta_frontera_recta(X, Y, theta):
    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()

    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max),
    np.linspace(x2_min, x2_max))

    h = sigmoid(np.c_[np.ones((xx1.ravel().shape[0], 1)),
    xx1.ravel(),
    xx2.ravel()].dot(theta))
    h = h.reshape(xx1.shape)

    # el cuarto parámetro es el valor de z cuya frontera se
    # quiere pintar
    plt.contour(xx1, xx2, h, [0.5], linewidths=2, colors='green')
    

def calcOptTheta(theta):
    result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, Y))
    return result[0]

def calcAciertos(X, Y, t):
    prediccion = 0 
    cont = 0
    aciertos = 0
    totales = len(Y)
    
    for i in X:
        if sigmoid(np.dot(i, t)) >= 0.5:
            prediccion = 1
        else:
            prediccion = 0
        
        if Y[cont] == prediccion:
            aciertos += 1

        cont += 1
            
    porcentaje = aciertos / totales * 100
    plt.text(82,100, str(porcentaje) + "% de aciertos")

def dibuja_puntos(X, Y):
    pos = np.where(Y == 1)
    plt.scatter(X[pos, 0], X[pos, 1], marker = '+', c = 'red')

    neg = np.where(Y == 0)
    plt.scatter(X[neg, 0], X[neg, 1], marker = '.', c = 'blue')

datos = carga_csv('ex2data1.csv')
X = datos[:, :-1]
np.shape(X)         

Y = datos[:, -1]
np.shape(Y)     

thetas = np.zeros(3)

dibuja_puntos(X, Y)

m = np.shape(X)[0]
n = np.shape(X)[1]
alpha = 0.01

# añadimos una columna de 1's a la X
X = np.hstack([np.ones([m, 1]), X])

nX = datos[:, :-1]
T = calcOptTheta(thetas)
pinta_frontera_recta(nX, Y, T)

calcAciertos(X, Y, T)

plt.show()