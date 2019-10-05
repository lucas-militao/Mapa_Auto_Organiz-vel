import numpy as np

def randomMatrix(linhas, colunas):
    return np.random.rand(linhas, colunas)

def zeroMatrix(linhas, colunas):
    return np.zeros(shape=(linhas, colunas))

def zeroVector(numberOfElements):
    return np.zeros(numberOfElements)
