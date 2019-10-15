import numpy as np
import math as mt
#Cálculo de distância euclidiana
def euclideanNorm(x, w):
    resultado = x - w
    resultado = np.power(resultado, 2)
    resultado = np.sum(resultado)
    resultado = np.sqrt(resultado)
    return resultado
#Cálculo do ajuste de pesos para o neurônio vencedor
def adjustWeightsWinner(w, n, x):
    resultado = w + n * (x - w)
    return resultado
#Cálculo do ajuste de pesos para raio = 1
def adjustWeightsR1(w, n, x):
    resultado = w + (n/2) * (x - w)
    return resultado
#Cálculo do operador de vizinhança
def __neighborOperator(w, wNeighbor, precisao):
    resultado = (w - wNeighbor)**2
    resultado = np.sum(resultado)
    resultado = np.sqrt(resultado)**2
    resultado = -resultado/(2*precisao**2)
    resultado = mt.exp(resultado)
    return resultado
#Cálculo do ajuste de pesos para raio > 1
def adjustWeightsRBiggerThan1(w, wNeighbor, n, x, precisao):
    neiOp = __neighborOperator(w, wNeighbor, precisao)
    resultado = wNeighbor + n * neiOp * (x - wNeighbor)
    return resultado
#Checagem de mudança significativa
def checkChange(w, wOld, precisao):
    changes = abs(w - wOld)

    for i in range(np.size(changes)):
        if(changes[i] > precisao):
            return False

    return True

def distanceFromTwoPoints(x1, y1, x2, y2):
    if not (x1 == x2 and y1 == y2):
        if x1 - x2 == 0 or y1 - y2 == 0:
            if x1 - x2 == 0:
                return abs(y1 - y2)
            elif y1 - y2 == 0:
                return abs(x1 - x2)
            else:
                return 0.0
        else:
            return np.hypot(abs(x1 - x2), abs(y1 - y2))
    else:
        return 0.0