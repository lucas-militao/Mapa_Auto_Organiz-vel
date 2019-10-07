import numpy as np

def euclideanNorm(x, w):
    resultado = x - w
    resultado = np.power(resultado, 2)
    resultado = np.sum(resultado)
    resultado = np.sqrt(resultado)
    return resultado


def adjustWeights(w, n, x):
    resultado = w + n * (x - w)
    return resultado