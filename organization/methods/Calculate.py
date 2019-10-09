import numpy as np

def euclideanNorm(x, w):
    resultado = x - w
    resultado = np.power(resultado, 2)
    resultado = np.sum(resultado)
    resultado = np.sqrt(resultado)
    return resultado

def adjustWeightsWinner(w, n, x):
    resultado = w + n * (x - w)
    return resultado

def adjustWeightsR1(w, n, x):
    resultado = w + (n/2) * (x - w)
    return resultado

def __neighborOperator(w, wNeighbor, precisao):
    resultado = np.exp(- (pow(np.abs(w - wNeighbor)),2)/(2*pow(precisao,2)))

def adjustWeightsRBiggerThan1(w, wNeighbor, n, x, precisao):
    neiOp = __neighborOperator(w, wNeighbor, precisao)
    resultado = wNeighbor + n * neiOp * (x - wNeighbor)
    return resultado

def checkChange(x, w, precisao):
    change = np.abs(x - w)
    change = np.min(change)

    if(change > precisao):
        return False
    # for i in range(np.size(change)):
    #     if(change[0,i] > precisao):
    #         return False
    return True




