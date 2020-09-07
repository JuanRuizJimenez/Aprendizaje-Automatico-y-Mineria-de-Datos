from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

#Función sigmoide
def sigmoide(x):
    s = 1 / (1 + np.exp(-x))
    return s

def calcAciertos(X, Y, t):
    #X = todas las X
    #Y = la Y de cada fila de X
    #t = cada fila de la matriz de thetas
    aciertos = 0
    totales = len(Y)
    dimThetas = len(t)

    for i in range(dimThetas):
        r = np.argmax(t[i]) + 1
        #print(str(r) + "------>" + str(Y[cont]))
        if(r==Y[i]):
            aciertos+=1     

    porcentaje = aciertos / totales * 100
    return porcentaje

# Selecciona aleatoriamente ejemplos y los pinta
def pinta_aleatorio(X):
    sample = np.random.choice(X.shape[0], 10)
    aux = X[sample, :].reshape(-1, 20)
    plt.imshow(aux.T)
    plt.axis("off")

def propagacion_hacia_delante(X, theta1, theta2):
    m = X.shape[0]
    a1 = np.hstack([np.ones([m, 1]), X])
    z2 = np.dot(a1, theta1.T)
    a2 = np.hstack([np.ones([m, 1]), sigmoide(z2)])
    z3 = np.dot(a2, theta2.T)
    h = sigmoide(z3)
    #return a1, z2, a2, z3, h
    return h

datos = loadmat ("ex3data1.mat")

#almacenamos los datos leídos en X e y
X = datos["X"]
Y = datos["y"]

yaux = np.ravel(Y) 

weights = loadmat("ex3weights.mat")
theta1, theta2 = weights["Theta1"], weights["Theta2"]
# Theta1 es de dimensión 25 x 401
# Theta2 es de dimensión 10 x 26

h = propagacion_hacia_delante(X, theta1, theta2)

result = calcAciertos(X, yaux, h)
text = str(result) + "% de aciertos."
print(text)

plt.axis("off")
plt.text(0,0, text)

#Mostramos los datos finalmente
plt.show()   