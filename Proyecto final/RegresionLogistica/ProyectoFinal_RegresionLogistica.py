from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import os
import scipy.io as sciOutput

#Función sigmoide
def sigmoide(x):
    s = 1 / (1 + np.exp(-x))
    return s

#Función para el coste
def coste(theta, X, Y, landa):
    H = sigmoide(np.matmul(X, theta))
    m = len(X)
    
    cost = ((- 1 / m) * (np.dot(Y, np.log(H)) + np.dot((1 - Y), np.log(1 - H)))) + ((landa / (2 * m)) * (np.sum(np.power(theta, 2))))
    
    return cost

#Función para calculo de gradiente
def gradiente(theta, XX, Y, landa):
    H = sigmoide(np.matmul(XX, theta))
    m=len(Y)
    grad = (1 / m) * np.matmul(XX.T, H - Y)
    
    aux=np.r_[[0],theta[1:]]

    firstPart = grad+(landa*aux/m)
    thetaAux = theta
    thetaAux[0] = 0

    result = firstPart + (landa / m * thetaAux)
    return result

def coste_y_gradiente(x0, X, Y, landa):
    return coste(x0,X,Y,landa), gradiente(x0, X, Y, landa)

def calcOptTheta(X, Y, maxIt, landa):
    result = opt.minimize(
        fun=coste_y_gradiente, 
        x0=np.zeros(X.shape[1]), 
        args=(X, Y, landa), 
        method='TNC', 
        jac=True, 
        options={'maxiter': maxIt})

    return result.x

def oneVsAll(X, y, num_etiquetas, reg, maxIt, landa):
    
    ThetasMatriz = np.zeros((num_etiquetas, X.shape[1]))

    i = 0
    while i < num_etiquetas:

        os.system('cls')
        print("Numero de etiquetas procesadas: ", i + 1, " de un total de ", num_etiquetas, " con lamda = ", landa, " y ", maxIt, " iteraciones.")
        auxY = (y == i).astype(int)
        ThetasMatriz[i, :] = calcOptTheta(X, auxY, maxIt, landa)
        i += 1

    return ThetasMatriz

def calcAciertos(X, Y, t):
    cont = 0
    aciertos = 0
    totales = len(Y)
    dimThetas = len(t)
    valores = np.zeros(dimThetas)

    for i in X:      
        p = 0
        for x in range(dimThetas):
            valores[p] = sigmoide(np.dot(i, t[x]))
            p+=1

        r = np.argmax(valores)

        if(r==Y[cont]):
            aciertos+=1     

        cont += 1

    porcentaje = aciertos / totales * 100
    return porcentaje

def pinta_aleatorio(X):
    sample = np.random.randint(low=0, high=len(X) - 1, size=1)
    aux = X[sample, :].reshape(-1, 20)
    plt.imshow(aux.T)
    plt.axis("off")

def runApplication():
    datos = loadmat("../proyecto_final_data_TRAIN.mat")

    #almacenamos los datos leídos en X e y
    X = datos["X"]
    y = datos["y"]

    yaux = np.ravel(y) 

    landas = [0.001, 0.01, 0.1, 1, 10, 50, 100, 300, 500]
    maxIterations = [70, 100, 150, 200, 300]
    num_labels = len(np.unique(yaux))

    testedThetasValues = dict()
    testedThetas = []
    p = 0

    aciertos = []
    myLandas = []
    myIter = []

    for i in landas:
        for r in maxIterations:
            landa = i
            one = (oneVsAll(X, yaux, num_labels, i, r, landa))
            testedThetasValues[p] = np.mean(one)
            testedThetas.append(one)

            myLandas.append(i)
            myIter.append(r)
            aciertos.append(calcAciertos(X, yaux, one))
            p += 1

    os.system('cls')

    val =  aciertos.index(max(aciertos)) #Lo mismo que max(testedThetasValues)

    print("Mejor porcentaje de acierto para entrenamiento: " ,str(aciertos[val]) + "% de acierto con un valor de lambda = ", myLandas[val], " con ", myIter[val], " iteraciones.")

    saveOutputData(myLandas, myIter, aciertos, testedThetas)

    return testedThetas

def saveOutputData(myLandas, myIter, myAciertos, thetas):
    dict = {
        "landas": myLandas,
        "iterations": myIter,
        "aciertos" : myAciertos,
        "thetas" : thetas
    }

    sciOutput.savemat("RegresionLogisticaOutput.mat", dict)

    print("Matriz guardada en ", "RegresionLogisticaOutput.mat")

    printOutputData()

def printOutputData():

    datos = loadmat("RegresionLogisticaOutput.mat")

    landas = datos["landas"]
    aciertos = datos["aciertos"]
    myIter = datos["iterations"]

    dibuja_puntos(landas, aciertos, "blue", "λ")

    dibuja_puntos(myIter, aciertos, "orange", "Iteraciones")

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

    text = "Porcentaje máximo de acierto para " + simbolo + " = " + str(round(q[np.argmax(q)], 2)) + "% con valor " + str(p[np.argmax(q)])
    ax.set_title(text)

    plt.show()

def calcula_aciertos_validacion():
    datos = loadmat("../proyecto_final_data_TRAIN.mat")
    datos2 = loadmat("RegresionLogisticaOutput.mat")
    #almacenamos los datos leídos en X e y
    Xval = datos["Xval"]
    yval = datos["yval"]

    yaux = np.ravel(yval)
    aciertos = []

    thetas = datos2["thetas"]

    for x in thetas:
        aciertos.append(calcAciertos(Xval, yaux, x))

    val =  aciertos.index(max(aciertos))

    print("Mejor porcentaje de acierto con los datos de validacion: ", str(aciertos[val]) + "%")

    return thetas[val]

def calcula_aciertos_test(thetaOpt):
    datos = loadmat("../proyecto_final_data_TRAIN.mat")

    Xtest = datos["Xtest"]
    ytest = datos["ytest"]

    ytest = np.ravel(ytest)

    print("Mejor porcentaje de acierto con los datos de test: ", str(calcAciertos(Xtest, ytest, thetaOpt)) + "%")

def main():
    #thetasEntrenamiento = runApplication()
    thetasOpt = calcula_aciertos_validacion()
    calcula_aciertos_test(thetasOpt)

main()


    