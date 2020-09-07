from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import time
import os

#Función sigmoide
def sigmoide(x):
    s = 1 / (1 + np.exp(-x))
    return s

def pinta_aleatorio(X):
    sample = np.random.choice(X.shape[0], 10)
    aux = X[sample, :].reshape(-1, 20)
    plt.imshow(aux.T)
    plt.axis("off")

#Función para el coste
def coste(theta1, theta2, X, Y, landa):
    H = propagacion_hacia_delante(X, theta1, theta2)
    m = len(X)
    cost = ((- 1 / m) * (np.dot(Y, np.log(H)) + np.dot((1 - Y), np.log(1 - H)))) # + ((landa / (2 * m)) * (np.sum(np.power(theta, 2))))
    print (cost)

#backprop devuelve el coste y el gradiente de una red neuronal de dos capas.    
def backprop(params_rn, num_entradas, num_ocultas, num_etiquetas, X, y, reg):
    theta1 = np.reshape(params_rn[:num_ocultas * (num_entradas + 1)], (num_ocultas, (num_entradas + 1)))
    theta2 = np.reshape(params_rn [num_ocultas * (num_entradas + 1 ):], (num_etiquetas, (num_ocultas + 1)))
    coste(theta1, theta2, X, y,reg)
    
    a1, z2, a2, z3, h = propagacion_hacia_delante(X, theta1, theta2)

    d3 = np.hstack([np.ones([m, 1]), sigmoide(z3)])

    d2 = np.dot(d3, theta2.T) * np.hstack([np.ones([m, 1]), sigmoide(z2)])

    #Hay que quitar la primera columna de delta2 sabe dios como

    delta1 = delta1 + d2 * a1.T

    delta2 = delta2 + d3 * a2.T


def propagacion_hacia_delante(X, theta1, theta2):
    m = X.shape[0]
    a1 = np.hstack([np.ones([m, 1]), X])
    z2 = np.dot(a1, theta1.T)
    a2 = np.hstack([np.ones([m, 1]), sigmoide(z2)])
    z3 = np.dot(a2, theta2.T)
    h = sigmoide(z3)
    return a1, z2, a2, z3, h
datos = loadmat ("ex4data1.mat")

#almacenamos los datos leídos en X e y
X = datos["X"]
Y = datos["y"]

weights = loadmat ("ex4weights.mat")
theta1, theta2 = weights["Theta1"], weights["Theta2"]
# Theta1 es de dimensión 25 x 401
# Theta2 es de dimensión 10 x 26

frame1 = """
   _________
   |       |
   |       |
 -------------
  ***    ***
   ^      ^
      ||
      --  
       
      ||
  """

frame2 = """
   _________
   |       |
   |       |
 -------------
  ***    ***
   ^      ^
      ||
      --  
       _
      |_|
  """
frame3 = """
   _________
   |       |
   |       |
 -------------
  ***    ***
   ^      ^
      ||
      --  
      __ 
     |__|
  """

frame4  = """
   _________
   |       |
   |       |
 -------------
  ***    ***
   ^      ^
      ||
      --  
     ____ 
    |____|  
           
  """

frame5  = """
   _________
   |       |
   |       |
 -------------
  ***    ***
   ^     ---
      ||
      --  
     ____    ________
    |____|  <__MUACK_|
           
  """

clear = lambda: os.system('cls') #on Windows System

while(1):
    clear()
    print(frame1)
    time.sleep(0.1)

    clear()
    print(frame2)
    time.sleep(0.1)

    clear()
    print(frame3)
    time.sleep(0.1)

    clear()
    print(frame4)
    time.sleep(0.1)

    clear()
    print(frame3)
    time.sleep(0.1)

    clear()
    print(frame2)
    time.sleep(0.1)

    clear()
    print(frame1)
    time.sleep(0.1)

    clear()
    print(frame5)
    time.sleep(2)
#yaux = np.ravel(Y)     

plt.show()