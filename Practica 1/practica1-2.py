import numpy as np
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt

def carga_csv(file_name):
    """carga el fichero csv especificado y lo
    devuelve en un array de numpy"""
    valores = read_csv(file_name, header=None).values
    # suponemos que siempre trabajaremos con float
    return valores.astype(float)

def normaliza_datos(X):

    numColumnas = X[1].size  
    
    media = np.zeros(numColumnas)
    varianza = np.zeros(numColumnas)

    for x in range(numColumnas):
        xData = X[:,x]
        media[x] = np.median(xData)
        varianza[x] = np.std(xData)

    X_normalizadas = (X - media) / varianza
     
    return X_normalizadas


def coste(X, Y, T):
    H = np.dot(X, T)
    Aux = (H - Y) ** 2
    return Aux.sum() / (2 * len(X))

def descenso_gradiente(X, Y, alpha, m, n):
    Thetas = np.zeros(X[1].size)

    for i in range(400):

        Thetas = Thetas - alpha * (1 / m) * ( X.T.dot(np.dot(X, Thetas) - Y) )

        coste(X, Y, Thetas)
        costes = []
        costes = np.append(costes, coste(X, Y, Thetas))

    return Thetas
    
def ecuacionNormal(X, Y):
     Thetas = np.zeros(X[1].size) 

     Xt = X.T
     aux = np.dot(Xt, X)
     aux2 = np.linalg.pinv(aux)
     aux3 = np.dot(aux2, Xt)
     Thetas = np.dot(aux3, Y)

     return Thetas


datos = carga_csv('ex1data2.csv')
X = datos[:, :-1]
np.shape(X)         

Y = datos[:, -1]
np.shape(Y)         

m = np.shape(X)[0]
n = np.shape(X)[1]
alpha = 0.01
nX = normaliza_datos(X)

# añadimos una columna de 1's a la nX
nX = np.hstack([np.ones([m, 1]), nX])

T1 = descenso_gradiente(nX, Y, alpha, m, n)

# añadimos una columna de 1's a la X
X = np.hstack([np.ones([m, 1]), X])

T2 = ecuacionNormal(X, Y)

print(T1)
print(T2)