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
def checkChange(x, w, precisao):
    change = np.abs(x - w)
    change = np.min(change)

    if(change > precisao):
        return False
    return True

    # for i in range(np.size(change)):
    #     if(change[0,i] > precisao):
    #         return False




