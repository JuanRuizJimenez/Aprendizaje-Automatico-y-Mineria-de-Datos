import numpy as np
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.io import loadmat
from checkNNGradients import checkNNGradients
from displayData import displayData

# Función sigmoide
def sigmoide(x):
    return 1 / (1 + np.exp(-x))

# Cálculo de la derivada de la función sigmoide
def derivada_sigmoide(x):
    return (sigmoide(x) * (1.0 - sigmoide(x)))

# Cáculo del coste no regularizado
def coste(m, h, y):
    #return ((1 / m) * (np.sum(np.dot(-Y.T, np.log(h)) - np.dot((1 - Y).T, np.log(1 - h)))))
    J = 0
    for i in range(m):
        J += np.sum(-y[i] * np.log(h[i]) \
             - (1 - y[i]) * np.log(1 - h[i]))
    return (J / m)

# Cálculo del coste regularizado
def costeRegularizado(m, h, Y, reg, theta1, theta2):
    return coste(m, h, Y) + ((reg / (2 * m)) * ((np.sum(np.square(theta1[:, 1:]))) + (np.sum(np.square(theta2[:,1:])))))    

#Devuelve la salida de la red neuronal así como los valores individuales
def PropagacionHaciaDelante(X, theta1, theta2):
    m = X.shape[0]
    a1 = np.hstack([np.ones([m, 1]), X])
    z2 = np.dot(a1, theta1.T)
    a2 = np.hstack([np.ones([m, 1]), sigmoide(z2)])
    z3 = np.dot(a2, theta2.T)
    h = sigmoide(z3)
    return a1, z2, a2, z3, h
    
# Devuelve el coste y el gradiente de una red neuronal de dos capas
def backprop(params_rn, num_entradas, num_ocultas, num_etiquetas, X, y, reg):    
    m = X.shape[0]

    # Sacamos ambas thetas
    theta1 = np.reshape(params_rn[:num_ocultas * (num_entradas + 1)],
            (num_ocultas, (num_entradas + 1)))

    theta2 = np.reshape(params_rn[num_ocultas * (num_entradas + 1): ], 
        (num_etiquetas, (num_ocultas + 1)))

    a1, z2, a2, z3, h = PropagacionHaciaDelante(X, theta1, theta2)  

    coste = costeRegularizado(m, h, y, reg, theta1, theta2) # Coste regularizado

    # Inicialización de dos matrices "delta" a 0 con el tamaño de los thethas respectivos
    delta1 = np.zeros_like(theta1)
    delta2 = np.zeros_like(theta2)

    # Por cada ejemplo
    for t in range(m):
        a1t = a1[t, :] # (1, 401)
        a2t = a2[t, :] # (1, 26)
        ht = h[t, :] # (1, 10)
        yt = y[t]

        d3t = ht - yt
        d2t = np.dot(theta2.T, d3t) * (a2t * (1 - a2t)) # (1, 26)

        delta1 = delta1 + np.dot(d2t[1:, np.newaxis], a1t[np.newaxis, :])
        delta2 = delta2 + np.dot(d3t[:, np.newaxis], a2t[np.newaxis, :])

    delta1 = delta1 / m
    delta2 = delta2 / m

    # Gradiente perteneciente a cada delta
    delta1[:, 1:] = delta1[:, 1:] + (reg * theta1[:, 1:]) / m
    delta2[:, 1:] = delta2[:, 1:] + (reg * theta2[:, 1:]) / m
    
    #Unimos los gradientes
    gradiente = np.concatenate((np.ravel(delta1), np.ravel(delta2)))
    
    return coste, gradiente

# Inicializa una matriz de pesos aleatorios
def pesosAleatorios(L_in, L_out):
    ini = 0.12
    out = np.random.uniform(low=-ini, high=ini, size=(L_out, L_in))
    out = np.hstack((np.ones((out.shape[0], 1)), out))
    return out

def calcAciertos(Y, h):
    aciertos = 0
    totales = len(Y)
    dimThetas = len(h)

    for i in range(dimThetas):
        r = np.argmax(h[i])
        if(r==Y[i]):
            aciertos+=1     

    porcentaje = aciertos / totales * 100
    return porcentaje

data = loadmat("ex4data1.mat")

y = data["y"].ravel()
X = data["X"]

num_entradas = X.shape[1]
capa_oculta = 25
num_labels = 10
landa = 1

lenY = len(y)
y = (y - 1)
y_onehot = np.zeros((lenY, num_labels))
for i in range(lenY):
    y_onehot[i][y[i]] = 1

#Calculo de pesos por una red ya entrenada
#weights = loadmat ("ex4weights.mat")
#Theta1, Theta2 = weights["Theta1"], weights["Theta2"]

#Inicialización de dos matrices de pesos de manera aleatoria
Theta1 = pesosAleatorios(400, 25) # (25, 401)
Theta2 = pesosAleatorios(25, 10) # (10, 26)

# Crea una lista de Thetas
Thetas = [Theta1, Theta2]

# Concatenación de las matrices de pesos en un solo vector
unrolled_Thetas = [Thetas[i].ravel() for i,_ in enumerate(Thetas)]
params = np.concatenate(unrolled_Thetas)

# Obtención de los pesos óptimos entrenando una red con los pesos aleatorios
optTheta = opt.minimize(fun=backprop, x0=params, 
        args=(num_entradas, capa_oculta, num_labels,
        X, y_onehot, landa), method='TNC', jac=True,
        options={'maxiter': 70})

#Calculo de la precision del gradiante gracias a checkNNGradients
print("Diferencia de precision de gradiantes: ", str(np.sum(checkNNGradients(backprop, 1))), ", maximo aceptado = 10e-9")

# Desglose de los pesos óptimos en dos matrices
Theta1Final = np.reshape(optTheta.x[:capa_oculta * (num_entradas + 1)],
    (capa_oculta, (num_entradas + 1)))

Theta2Final = np.reshape(optTheta.x[capa_oculta * (num_entradas + 1): ], 
    (num_labels, (capa_oculta + 1)))

# H, resultado de la red al usar los pesos óptimos
a1, z2, a2, z3, h = PropagacionHaciaDelante(X, Theta1Final, Theta2Final) 

# Cálculo de la precisión de la red neuronal
print("{0:.2f}% de precision".format(calcAciertos(y,h)))
