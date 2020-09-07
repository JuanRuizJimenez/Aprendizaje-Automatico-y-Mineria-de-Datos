import numpy as np
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt

def carga_csv(file_name):
    """carga el fichero csv especificado y lo
    devuelve en un array de numpy"""
    valores = read_csv(file_name, header=None).values
    # suponemos que siempre trabajaremos con float
    return valores.astype(float)

def coste(X, Y, T):
    H = np.dot(X, T)
    Aux = (H - Y) ** 2
    return Aux.sum() / (2 * len(X))

def descenso_gradiente(X, Y, alpha, m, n):
    Thetas = np.zeros(2)
    i = 0

    while (i < 2500):

        Thetas = Thetas - alpha * (1 / m) * ( X.T.dot(np.dot(X, Thetas) - Y) )

        coste(X, Y, Thetas)
        costes = []
        costes = np.append(costes, coste(X, Y, Thetas))
        print(costes)

        i = i + 1

    pintaRecta(Thetas, X, Y)

def pintaRecta(T, x, y):
        x =  np.arange(np.min(X[:, 1]), np.max(X[:, 1]), 0.1)
        y = x.copy()

        a = len(x)
        i = 0

        while (i < a):
                y[i] = (x[i] * T[1]) + T[0]
                i = i + 1
        
        plt.plot(x, y, c='blue')

def pintaPuntosFuncion(X, Y):
    np.linspace(X[:, 1], Y, 256)
    plt.scatter(X[:, 1], Y, c='red', marker='x' )
    plt.show()

datos = carga_csv('ex1data1.csv')
X = datos[:, :-1]
np.shape(X)         # (97, 1)
Y = datos[:, -1]
np.shape(Y)         # (97,)
m = np.shape(X)[0]
n = np.shape(X)[1]

# añadimos una columna de 1's a la X
X = np.hstack([np.ones([m, 1]), X])
alpha = 0.01

descenso_gradiente(X, Y, alpha, m, n)
pintaPuntosFuncion(X, Y)