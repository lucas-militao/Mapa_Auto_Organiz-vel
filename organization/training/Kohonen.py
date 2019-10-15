import numpy as np

from organization.methods import Calculate as cal
from functions import MatrixGenerator as mg

def __operacao(x, w):

    dist = mg.zeroVector(np.size(w[:,0]))

    for i in range(np.size(dist)):
        dist[i] = cal.euclideanNorm(x, w[i])

    vencedor = np.argmin(dist)

    return vencedor


def __unidimensional(qtdNeuronios, entradas, pesos, taxaAprendizagem, precisao, raio):
    N = qtdNeuronios
    x = entradas
    w = pesos
    n = taxaAprendizagem
    r = raio
    epoca = 0

    pare1 = False
    pare2 = False
    pare3 = False

    # inicializa o vetor que irá receber as distâncias euclidianas
    dist = mg.zeroVector(N)

    while(epoca < 1000 and (pare1 != True or pare2 != True or pare3 != True)):

        #calcula as distancias euclidianas
        for i in range(np.size(dist)):
            dist[i] = np.linalg.norm(x - w[i])
            # dist[i] = cal.euclideanNorm(x, w[i])
        #definir vencedor
        vencedor = np.argmin(dist)
        wOld = w[vencedor]
        #ajustar pesos do vencedor
        w[vencedor] = cal.adjustWeightsWinner(w[vencedor], n, x)
        if(r > 0):
            #Para r = 1, será realizado o ajuste utilizando apenas a expressão com n/2
            if(r == 1):
                try:
                    wOld = np.hstack(w[vencedor + r])
                    wOld = np.hstack(w[vencedor - r])
                    w[vencedor + r] = cal.adjustWeightsR1(w[vencedor + r], n, x)
                    pare1 = cal.checkChange(w[vencedor + r], wOld, precisao)
                    w[vencedor - r] = cal.adjustWeightsR1(w[vencedor - r], n, x)
                    pare2 = cal.checkChange(w[vencedor - r], wOld, precisao)
                except:
                    print()
            #Para r > 1, será realizado o ajuste dos vizinhos utilizando as duas expressões
            elif(r > 1):
                try:
                    for j in range(r):
                        wOld = np.hstack(w[vencedor + r])
                        wOld = np.hstack(w[vencedor - r])
                        w[vencedor + r] = cal.adjustWeightsRBiggerThan1(w, w[vencedor + r], n, x, precisao)
                        pare1 = cal.checkChange(w[vencedor + r], wOld, precisao)
                        w[vencedor - r] = cal.adjustWeightsRBiggerThan1(w, w[vencedor - r], n, x, precisao)
                        pare2 = cal.checkChange(w[vencedor - r], wOld, precisao)
                except:
                    print()

        #verificar se houve mudança significativa
        pare3 = cal.checkChange(w[vencedor], wOld, precisao)

        epoca += 1
    print(epoca)
    return w

def trainUnidimensional (qtdNeuronios, entradas, taxaAprendizagem, precisao, raio):
    N = qtdNeuronios
    x = entradas
    n = taxaAprendizagem
    r = raio

    #inicializando a matriz de pesos com valores entre 1 e -1
    w = mg.randomVectorHighLow(1, -1, N, np.size(x[0,:]))
    wOld = mg.randomVectorHighLow(1, -1, N, np.size(x[0,:]))

    #treinamento
    for i in range(np.size(x[:,0])):
        w = __unidimensional(N, x[i], w, n, precisao, r)
    #resultado
    amostras = ["A1", "B1", "C1", "D1", "E1", "J1", "K1",
                "A2", "B2", "C2", "D2", "E2", "J2", "K2",
                "A3", "B3", "C3", "D3", "E3", "J3", "K3"]
    resultado = mg.emptyStringVector(N)

    for i in range(np.size(x[:,0])):
        v = __operacao(x[i], w)
        if(resultado[v] == ''):
            resultado[v] = amostras[i]
        elif(resultado[v] != ''):
            resultado[v] += " {}" .format(amostras[i])
    print(resultado)

        # print("para a amostra {}" .format(i) +
        #       " o neuronio foi: {}" .format(v))

def mapaKohonen(qtdNeuronio):
    qtd = int(qtdNeuronio**(1/2))
    matrix = mg.zeroMatrix(qtd,qtd)
    indice = 0
    for i in range(np.size(matrix[:,0])):
        for j in range(np.size(matrix[0,:])):
            matrix[i,j] = indice
            indice += 1
    return matrix

def neighborhood(linhas, colunas, raio):
    vizinhos = np.array([[]])



















