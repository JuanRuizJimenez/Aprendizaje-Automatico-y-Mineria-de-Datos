import numpy as np
from scipy.io import loadmat
import scipy.optimize as opt
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def sigmoide(x):
    return 1 / (1 + np.exp(-x))

#Función para el coste
def coste(theta, X, y, landa, m):
    h = np.dot(X, theta)

    return ((1 / (2 * m)) * (np.sum(np.square(h - y)))) + ((landa / (2 * m)) * np.sum(np.square(theta[1:len(theta)])))
    
#Función para calculo de gradiente
def gradiente(theta, XX, Y, landa, m): 
    h = np.dot(XX, theta)

    grad = (1 / m) * np.dot(XX.T, h - Y)+((landa/m) * theta)
    return grad

def coste_y_gradiente(theta, X, Y, landa, m):

    theta = theta.reshape(-1, Y.shape[1])

    cost = coste(theta, X, Y, landa, m) 
    grad = gradiente(theta, X, Y, landa, m)
    
    grad[0] = (1 / m) * np.dot(X.T, np.dot(X, theta) - Y)[0]

    return (cost, grad.flatten())

def calcOptTheta(X, Y, landa):
    theta = np.zeros((X.shape[1], 1))
    
    def costFunction(theta):
        return coste_y_gradiente(theta, X, Y, landa, len(X))

    result = minimize(fun=costFunction, x0=theta, method='CG', jac=True, options={'maxiter':200})
    
    return result.x

def curva_aprendizaje(X, y, landa, Xval, yval):

    err1 = np.zeros((len(X)))
    err2 = np.zeros((len(X)))

    i = 1
    while (i < len(X) + 1):
        thetas = calcOptTheta(X[0:i], y[0:i], landa)

        err1[i - 1] = coste_y_gradiente(thetas, X[0:i], y[0:i], landa, len(X))[0]
        err2[i - 1] = coste_y_gradiente(thetas, Xval, yval, landa, len(Xval))[0]
        i += 1   

    return err1, err2    

def pinta_puntos(X, Y):
    plt.scatter(X, Y, 100,  marker = 'x', c = 'red', label = 'Entrada')

def pinta_Curva_Aprendizaje(err1, err2):
    
    a = np.arange(len(err1))
    b = err1
    plt.plot(a, b, c="blue", label="Train")

    d = err2[0:len(err1)]
    plt.plot(a, d, c="orange", label="Cross Validation")

def normaliza_Matriz(X):
    mu = np.mean(X, axis=0)
    X_norm = X - mu
    
    sigma = np.std(X_norm, axis=0)
    X_norm = X_norm / sigma
    
    return X_norm, mu, sigma

def transforma_entrada(X, p):
    nX = X
    for i in range(1, p):
        nX = np.column_stack((nX, np.power(X, i+1)))   

    return nX
    
def pinta_regresion_Polinomial(X, p, mu, sigma, theta):
    x = np.array(np.arange(min(X) - 5,  max(X) + 6, 0.02))
    nX = transforma_entrada(x, p)
    nX = nX - mu
    nX = nX / sigma
    nX = np.insert(nX, 0, 1, axis=1)
    plt.plot(x, np.dot(nX, theta))

def main():
    data = loadmat("ex5data1.mat")

    X = data["X"]
    y = data["y"]
    Xval = data["Xval"]
    yval = data["yval"]

    landa = 0
    p = 8

    nuevaentrada = transforma_entrada(X, p)
    nuevaentrada, mu, sigma = normaliza_Matriz(nuevaentrada)
    nuevaentrada = np.insert(nuevaentrada, 0, 1, axis=1)  
    
    thetaOpt = calcOptTheta(nuevaentrada, y, landa)
    pinta_regresion_Polinomial(X, p, mu, sigma, thetaOpt)
    
    pinta_puntos(X,y)
    plt.show()

    neuvaEntradaValidacion = transforma_entrada(Xval, p)
    neuvaEntradaValidacion = neuvaEntradaValidacion - mu
    neuvaEntradaValidacion = neuvaEntradaValidacion / sigma
    neuvaEntradaValidacion = np.insert(neuvaEntradaValidacion, 0, 1, axis=1)

    err1, err2 = curva_aprendizaje(nuevaentrada, y, landa, neuvaEntradaValidacion, yval)

    pinta_Curva_Aprendizaje(err1, err2)

    plt.show()

main()