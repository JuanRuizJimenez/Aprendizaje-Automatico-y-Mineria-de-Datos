import numpy as np
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.io import loadmat
from checkNNGradients import checkNNGradients
from displayData import displayData
import scipy.io as sciOutput
import os
import time

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

    print(totales)
    print(dimThetas)

    for i in range(dimThetas):
        r = np.argmax(h[i])
        if(r==Y[i]):
            aciertos+=1     

    porcentaje = aciertos / totales * 100
    return porcentaje

def runApplication():

    os.system('cls')
    print("Comenzando a trabajar...")
    tic = time.time()
    data = loadmat("../proyecto_final_data_TRAIN.mat")

    y = data["y"].ravel()
    X = data["X"]

    num_entradas = X.shape[1]
    num_labels =  len(np.unique(y))

    landas = [0.01, 0.1, 1, 10, 50, 100]
    maxIterations = [70, 100, 200]
    numCapasOcultas = [25, 75, 125]

    aciertos = []
    myLandas = []
    myIter = []
    myOcultas = []
    thetas = []

    lenY = len(y)
    y_onehot = np.zeros((lenY, num_labels))
    for i in range(lenY):
        y_onehot[i][y[i]] = 1

    tam = len(landas) * len(maxIterations) * len(numCapasOcultas)
    aux = 1

    for l in landas:
        for mi in maxIterations:
            for numC in numCapasOcultas:
                
                #Inicialización de dos matrices de pesos de manera aleatoria
                Theta1 = pesosAleatorios(len(X[0]), numC)
                Theta2 = pesosAleatorios(numC, num_labels) 

                # Crea una lista de Thetas
                Thetas = [Theta1, Theta2]

                # Concatenación de las matrices de pesos en un solo vector
                unrolled_Thetas = [Thetas[i].ravel() for i,_ in enumerate(Thetas)]
                params = np.concatenate(unrolled_Thetas)

                # Obtención de los pesos óptimos entrenando una red con los pesos aleatorios
                optTheta = opt.minimize(fun=backprop, x0=params, 
                        args=(num_entradas, numC, num_labels,
                        X, y_onehot, l), method='TNC', jac=True,
                        options={'maxiter': mi})

                #Calculo de la precision del gradiante gracias a checkNNGradients
                #print("Diferencia de precision de gradiantes: ", str(np.sum(checkNNGradients(backprop, 1))), ", maximo aceptado = 10e-9")

                # Desglose de los pesos óptimos en dos matrices
                Theta1Final = np.reshape(optTheta.x[:numC * (num_entradas + 1)],
                    (numC, (num_entradas + 1)))

                Theta2Final = np.reshape(optTheta.x[numC * (num_entradas + 1): ], 
                    (num_labels, (numC + 1)))

                # H, resultado de la red al usar los pesos óptimos
                a1, z2, a2, z3, h = PropagacionHaciaDelante(X, Theta1Final, Theta2Final) 

                aciertos.append(calcAciertos(y,h))
                myLandas.append(l)
                myIter.append(mi)
                myOcultas.append(numC)
                thetas.append(h)

                os.system('cls')
                print("Num de datos procesados: " , aux ," de un total " , tam, " combinaciones.")
                aux+=1

    val =  aciertos.index(max(aciertos))

    saveOutputData(myLandas, myIter, aciertos, myOcultas, thetas)

    print("Mejor porcentaje de acierto: " ,str(aciertos[val]) + "% de acierto con un valor de lambda = ", myLandas[val], " con ", myIter[val], " iteraciones y ", myOcultas[val] , " capas ocultas.")
    toc = time.time()

    print("Tiempo empleado para entrenar el sistema: ", round((toc - tic) / 60.), " minutos, ", (toc - tic), " segundos.")

    printOutputData()

def saveOutputData(myLandas, myIter, myAciertos, myOcultas, thetas):
    dict = {
        "landas": myLandas,
        "iterations": myIter,
        "aciertos" : myAciertos,
        "capasOcultas" : myOcultas,
        "thetas" : thetas
    }

    sciOutput.savemat("RedesNeuronalesOutput.mat", dict)

    print("Matriz guardada en ", "RedesNeuronalesOutput.mat")

def  printOutputData():
    datos = loadmat("RedesNeuronalesOutput.mat")

    landas = datos["landas"]
    aciertos = datos["aciertos"]
    myIter = datos["iterations"]
    myOcultas = datos["capasOcultas"]

    dibuja_puntos(landas, aciertos, "blue", "λ")

    dibuja_puntos(myIter, aciertos, "orange", "Iteraciones")

    dibuja_puntos(myOcultas, aciertos, "green", "Capas Ocultas")

def dibuja_puntos(X, Y, color, simbolo):
    
    combined = np.vstack((X, Y)).T

    differentValues = np.unique(combined[:,0])

    p = []
    q = []

    for x in differentValues:
        arr = np.argwhere(combined==x)
        maxiItem = np.max(arr)
        p.append(combined[maxiItem,0])
        q.append(combined[maxiItem,1])

    texto = "Porcentaje máximo del valor " + simbolo

    fig = plt.figure()
    fig.suptitle("Porcentaje de acierto máximo para " + simbolo, fontsize=14, fontweight='bold')

    ax = fig.add_subplot(111)
    fig.subplots_adjust(top=0.85)
    
    plt.scatter(p, q, c="red", label=texto)
    plt.plot(p, q, c="blue")
    plt.legend()

    text = "Porcentaje máximo de acierto para " + simbolo + "= " + str(round(q[np.argmax(q)], 2)) + "% con valor: " + str(p[np.argmax(q)])
    ax.set_title(text)

    plt.show()

def calcula_aciertos_validacion():
    datos = loadmat("../proyecto_final_data_TRAIN.mat")
    datos2 = loadmat("RedesNeuronalesOutput.mat")

    yval = datos["yval"]

    yaux = np.ravel(yval)
    aciertos = []

    thetas = datos2["thetas"]

    print(len(yaux))

    for x in thetas:
        print(len(x))
        aciertos.append(calcAciertos(yaux, x))

    val =  aciertos.index(max(aciertos))

    print("Mejor porcentaje de acierto con los datos de validacion: ", str(aciertos[val]) + "%")

    return thetas[val]

def calcula_aciertos_test(thetaOpt):
    datos = loadmat("../proyecto_final_data_TRAIN.mat")

    ytest = datos["ytest"]

    ytest = np.ravel(ytest)

    print("Mejor porcentaje de acierto con los datos de test: ", str(calcAciertos(ytest, thetaOpt)) + "%")

def main():
    #runApplication()
    thetaOpt = calcula_aciertos_validacion()
    calcula_aciertos_test(thetaOpt)
main()    