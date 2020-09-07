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

    err1 = np.zeros((len(landa)))
    err2 = np.zeros((len(landa)))

    i = 0
    while (i < len(landa)):
        thetas = calcOptTheta(X, y, landa[i])

        #IMPORTANTE que landa tiene que ser 0 aquí 
        err1[i] = coste_y_gradiente(thetas, X, y, 0, len(X))[0]
        err2[i] = coste_y_gradiente(thetas, Xval, yval, 0, len(Xval))[0]
        i += 1   

    return err1, err2    

def pinta_Curva_Aprendizaje(landaV, err1, err2):
    b = err1
    plt.plot(landaV, b, c="blue", label="Train")

    d = err2
    plt.plot(landaV, d, c="orange", label="Cross Validation")

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

def main():
    data = loadmat("ex5data1.mat")

    X = data["X"]
    y = data["y"]
    Xval = data["Xval"]
    yval = data["yval"]
    Xtest = data["Xtest"]
    ytest = data["ytest"]

    landas = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])
    p = 8

    nuevaentrada = transforma_entrada(X, p)
    nuevaentrada, mu, sigma = normaliza_Matriz(nuevaentrada)
    nuevaentrada = np.insert(nuevaentrada, 0, 1, axis=1)  
    
    neuvaEntradaValidacion = transforma_entrada(Xval, p)
    neuvaEntradaValidacion = neuvaEntradaValidacion - mu
    neuvaEntradaValidacion = neuvaEntradaValidacion / sigma
    neuvaEntradaValidacion = np.insert(neuvaEntradaValidacion, 0, 1, axis=1)

    err1, err2 = curva_aprendizaje(nuevaentrada, y, landas, neuvaEntradaValidacion, yval)  

    pinta_Curva_Aprendizaje(landas, err1, err2)
    

    #De todo esto anterior, sacamos como conclusión que el valor óptimo para lambda
    # es 3, por lo que ahora, con los ejemplos de test comprobaremos el error obtenido
    landa = 3
    
    neuvaEntradaTest = transforma_entrada(Xtest, p)
    neuvaEntradaTest = neuvaEntradaTest - mu
    neuvaEntradaTest = neuvaEntradaTest / sigma
    neuvaEntradaTest = np.insert(neuvaEntradaTest, 0, 1, axis=1)

    optTheta = calcOptTheta(nuevaentrada, y, landa)

    #Lambda siempre = 0
    error_test = coste_y_gradiente(optTheta, neuvaEntradaTest, ytest, 0, len(neuvaEntradaTest))[0] #[0] para que lo que devuelva sea el coste

    print("Error obtenido de: ", error_test)
   
    plt.show()

main()