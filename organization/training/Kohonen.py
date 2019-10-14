import numpy as np

from organization.methods import Calculate as cal
from functions import MatrixGenerator as mg

def __operacao(x, w):

    dist = mg.zeroVector(np.size(w[:,0]))

    for i in range(np.size(dist)):
        dist[i] = cal.euclideanNorm(x, w[i])

    vencedor = np.argmin(dist)

    return vencedor

# def __faseOperacao(x, w):
#     dist = mg.zeroVector(np.size(w[:, 0]))
#
#     for i in range(np.size(dist)):
#         dist[i] = cal.euclideanNorm(x, w[i])
#
#     vencedor = np.argmin(dist)


def __unidimensional(qtdNeuronios, entradas, pesos, taxaAprendizagem, precisao, raio):
    N = qtdNeuronios
    x = entradas
    w = pesos
    n = taxaAprendizagem
    r = raio
    epoca = 0
    pare = False

    # inicializa o vetor que irá receber as distâncias euclidianas
    dist = mg.zeroVector(N)

    while(epoca < 100000 and pare != True):

        #calcula as distancias euclidianas
        for i in range(np.size(dist)):
            dist[i] = cal.euclideanNorm(x, w[i])
        #definir vencedor
        vencedor = np.argmin(dist)
        #ajustar pesos do vencedor
        w[vencedor] = cal.adjustWeightsWinner(w[vencedor], n, x)
        if(r > 0):
            #Para r = 1, será realizado o ajuste utilizando apenas a expressão com n/2
            if(r == 1):
                try:
                    w[vencedor + r] = cal.adjustWeightsR1(w[vencedor + r], n, x)
                    w[vencedor - r] = cal.adjustWeightsR1(w[vencedor - r], n, x)
                except:
                    print()
            #Para r > 1, será realizado o ajuste dos vizinhos utilizando as duas expressões
            elif(r > 1):
                try:
                    for j in range(r):
                        w[vencedor + r] = cal.adjustWeightsRBiggerThan1(w, w[vencedor + r], n, x, precisao)
                        w[vencedor - r] = cal.adjustWeightsRBiggerThan1(w, w[vencedor - r], n, x, precisao)
                except:
                    print()

        #verificar se houve mudança significativa
        pare = cal.checkChange(x, w[vencedor], precisao)

        epoca += 1
    # print(epoca)
    return w

def trainUnidimensional (qtdNeuronios, entradas, taxaAprendizagem, precisao, raio):
    N = qtdNeuronios
    x = entradas
    n = taxaAprendizagem
    r = raio

    epoca = 0
    pare = False

    #inicializando a matriz de pesos com valores entre 1 e -1
    w = mg.randomVectorHighLow(1, -1, N, np.size(x[0,:]))

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

def mapaKohonen(numeroNeuronios):
    tam = int(numeroNeuronios ** (1/2))
    matriz = mg.zeroMatrix(tam, tam)
    count = 0
    for i in range(tam):
        for j in range(tam):
            matriz[i,j] = count
            count += 1
    return matriz

def bidimensional(numeroNeuronios, pesos, entradas, taxaAprendizagem, raio, precisao):
    N = numeroNeuronios
    w = pesos
    x = entradas
    n = taxaAprendizagem
    r = raio
    p = precisao
    parar = False

    dist = mg.zeroVector(N)

    epoca = 0

    while(epoca < 100 and parar == False):
        # calcula as distancias euclidianas
        for i in range(np.size(dist)):
            dist[i] = cal.euclideanNorm(x, w[i])
        # definir vencedor
        vencedor = np.argmin(dist)
        # ajustar pesos do vencedor
        w[vencedor] = cal.adjustWeightsWinner(w[vencedor], n, x)
        # ajustar pesos dos vizinhos
        if (r == 1):
            vizinhos = np.array((vencedor - 3, vencedor + 1, vencedor - 1, vencedor + 3))
            for i in range(vizinhos.size):
                try:
                    w[vizinhos[i]] = cal.adjustWeightsR1(w[vizinhos[i]], n, x)
                except:
                    print()
        elif (r > 1):
            vizinhos = np.array((vencedor + 1, vencedor - 1,
                                 vencedor + 2, vencedor - 2,
                                 vencedor + 3, vencedor - 3,
                                 vencedor + 4, vencedor - 4))
            for i in range(vizinhos.size):
                try:
                    w[vizinhos[i]] = cal.adjustWeightsRBiggerThan1(w[vencedor], w[vizinhos[i]], n, x, precisao)
                except:
                    print()

            parar = cal.checkChange(x, w[vencedor], precisao)
            epoca += 1

        print(epoca)
        return w





def trainBidimensional(numeroNeuronios, entradas, taxaAprendizagem, raio, precisao):
    N = numeroNeuronios
    x = entradas
    n = taxaAprendizagem
    r = raio
    p = precisao

    w = mg.randomVectorHighLow(1, -1, N, np.size(x[0,:]))
    mapa = mapaKohonen(N)


    for i in range(np.size(x[:,0])):
        w = bidimensional(numeroNeuronios, w, x[i], taxaAprendizagem, r, p)

    # print(w)



















