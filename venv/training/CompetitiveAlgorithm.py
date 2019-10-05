import numpy as np
from functions import MatrixGenerator as mg

def euclideanNorm(x, w):
    resultado = x - w
    resultado = np.power(resultado, 2)
    resultado = np.sum(resultado)
    resultado = np.sqrt(resultado)
    return resultado

def adjustWeights(w, n, x):
    resultado = w + n * (x - w)
    return resultado

def train(entradas, pesos, taxaAprendizagem):

    epoca = 0
    precisao = 0,1
    finish = False
    dist = mg.zeroVector(np.size(pesos[:,0]))
    menorDistancia = 0

    while(epoca < 100 and epoca == False):

        for i in range(np.size(entradas[:,0])):
            for j in range(np.size(pesos[:,0])):

                dist[j] = euclideanNorm(entradas[i], pesos[j])

            menorDistancia = np.argmin(dist)

            pesos[menorDistancia, :] = adjustWeights(pesos[menorDistancia], taxaAprendizagem, entradas[i])

        # epoca += 1
        epoca = 100

    return pesos