import process_email 
import codecs
import get_vocab_dict as vocab
import numpy as np
import os
from sklearn import svm

def read_file(path):

    print("Leyendo archivos de ", path)

    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if '.txt' in file:
                files.append(os.path.join(r, file))

    j = 0
    vocab_dict = vocab.getVocabDict()

    arrayDict = np.array(list(vocab_dict.items()))

    X = np.zeros((len(files), len(arrayDict)))

    for f in files:
        email_contents = codecs.open(f, "r", encoding="utf−8", errors="ignore").read()

        email = process_email.email2TokenList(email_contents)

        aux = np.zeros(len(arrayDict))

        for i in range(len(email)):
            index = np.where(arrayDict[:, 0] == email[i])
            aux[index] = 1

        X[j] = aux
        j = j + 1

    print("Archivos de ", path, "leídos y guardados en X.")
    return X

def main():
    X = read_file('spam/')
    y = np.ones(len(X)) #Los que sabemos que son spam los marcamos como 1 en la salida

    XnoSpam = read_file('easy_ham/')
    yNoSpam = np.zeros(len(XnoSpam)) #Los que sabemos que NO son spam los marcamos como 0 en la salida

    #Juntamos los que son spam y los que no en las X's e Y's
    X = np.concatenate((X, XnoSpam))
    y = np.concatenate((y, yNoSpam))

    pinta_puntos(X, y)
    plt.show()

    print("Entrenando la red...")

    C = 0.1
    yravel = y.ravel()
    svc = svm.SVC(C, 'linear')
    svc.fit(X, yravel)
    p = svc.predict(X)

    print('Encontramos que la red tiene una tasa de acierto del: {0:.2f}%'.format(np.mean((p == yravel).astype(int)) * 100))

    print('Añadimos mas ejemplos de entrenamiento...')

    xnoHardSpam = read_file('hard_ham/')
    ynoHardSpam = np.zeros(len(xnoHardSpam))

    X = np.concatenate((X, xnoHardSpam))
    y = np.concatenate((y, ynoHardSpam))

    C = 0.1
    yravel = y.ravel()
    svc = svm.SVC(C, 'linear')
    svc.fit(X, yravel)
    p = svc.predict(X)

    print('Encontramos que la red tiene una tasa de acierto final del: {0:.2f}%'.format(np.mean((p == yravel).astype(int)) * 100))

main()