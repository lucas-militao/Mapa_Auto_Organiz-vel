import numpy as np

def randomMatrix(linhas, colunas):
    return np.random.rand(linhas, colunas)

def zeroMatrix(linhas, colunas):
    return np.zeros((linhas, colunas))

def zeroVector(numberOfElements):
    return np.zeros(numberOfElements)

def randomVectorHighLow(high, low, linhas, colunas):
    return ( np.random.rand(linhas, colunas) * (high - low) ) + low
    # return high * np.random.random_sample((linhas, colunas)) + low
    # return np.random.randint(low, high, size=(linhas, coluna))

