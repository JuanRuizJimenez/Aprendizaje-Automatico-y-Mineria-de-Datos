import PIL
from PIL import Image
import os
import shutil
import time
import numpy as np
from numpy import genfromtxt
import scipy.io as sciOutput

def transform_images_inside_path(path, outputName):

    paths = ["airplane/", "car/", "cat/", "dog/", "flower/", "fruit/", "motorbike/", "person/"]

    matEntrenamientoX = []
    matEntreanmientoY = []

    matValidacionX = []
    matValidacionY = []

    matTestX = []
    matTestY = []

    y = np.array(8) #airplane, car, cat, dog, flower, fruit, motorbike, person
    y = {0,1,2,3,4,5,6,7}
    
    #Cogemos todos los archivos jpg que encontremos dentro de la carpeta especificada
    for concretePath in paths:
        files = []
        for r, d, f in os.walk(path + concretePath):
            for file in f:
                if '.jpg' in file and not "_resized" in file:
                    files.append(file)

        #Comprobamos que el path no exista y si es así, lo eliminamos por completo para no tener mal los datos
        if os.path.exists(path + concretePath + "Resized/"):
            shutil.rmtree(path + concretePath + "Resized/")    
        
        #Creamos de nuevo el directorio que hemos borrado
        os.mkdir(path + concretePath + "Resized/")

        totalFiles = len(files)
        procesImages = 1

        for f in files:
            #BUSCAR EN Y EL NOMBRE DE LA IMAGEN Y COGER EL NOMBRE DE LA BALLENA ASOCIADO PARA SACAR LA Y FINAL
            os.system("cls")
        
            img = Image.open(path + concretePath + f)
            img = img.resize((20, 20), Image.ANTIALIAS) #Reescalamos la imagen
            #img.save(path + "Resized/" + f[:-4] + "_resized.jpg")
            
            processedImage = True
            values = []
            pix  = img.load()
            for i in range(20):
                for j in range(20): 
                    rgb = pix[i,j]
                    try:
                        rgbInteger = (int)(("%02x%02x%02x"%rgb), 16)
                        values.append(rgbInteger)
                    except:
                        processedImage = False

            if(processedImage):              
                aux = paths.index(concretePath)

                #Dividir en entrenamiento 60%, validación 20% y test 20%
                if procesImages - 1 < (int)(0.2 * totalFiles):
                    matTestX.append(values)
                    matTestY.append(aux)

                elif procesImages - 1 < (int)(0.4 * totalFiles):
                    matValidacionX.append(values)
                    matValidacionY.append(aux)

                else:
                    matEntrenamientoX.append(values)
                    matEntreanmientoY.append(aux)

                print("Se han procesado ", procesImages, " imagenes de un total de ", totalFiles, " Carpeta: ", (aux + 1), "/", (len(paths)))
                procesImages += 1

    X = np.array(matEntrenamientoX)
    y = np.array(matEntreanmientoY)

    dict = {
        "X": X,
        "y": y,
        "Xval" : matValidacionX,
        "yval" : matValidacionY,
        "Xtest": matTestX,
        "ytest" : matTestY
    }

    sciOutput.savemat(outputName, dict)

    print("Matriz guardada en ", outputName)

    #Una vez hemos terminado de procesar los datos, podemos borrar la carpeta con imágenes adicionales que hemos creado
    if os.path.exists(path + "Resized/"):
       shutil.rmtree(path + "Resized/")    


def main():

    tic = time.time()
    transform_images_inside_path("Images/", "proyecto_final_data_TRAIN.mat")

    toc = time.time()
    
    print("Tiempo empleado para modificar la entrada al formato necesario: ", round((toc - tic) / 60.), " minutos, ", (toc - tic), " segundos.")

main()