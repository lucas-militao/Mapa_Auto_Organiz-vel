import numpy as np
from functions import Normalization as nm
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

    x = nm.normalization(entradas)
    w = nm.normalization(pesos)

    dist = mg.zeroVector(np.size(pesos[:,0]))
    menorDistancia = 0

    epoca = 0

    while(epoca < 5):

        for i in range(np.size(x[:,0])):
            for j in range(np.size(w[:,0])):

                dist[j] = euclideanNorm(x[i], w[j])

            menorDistancia = np.argmin(dist)

            w[menorDistancia, :] = adjustWeights(w[menorDistancia], taxaAprendizagem, x[i])
            w[menorDistancia, :] = nm.normalization(w[menorDistancia, :])

        epoca += 1

    return w