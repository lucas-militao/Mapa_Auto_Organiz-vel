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
    print(epoca)
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
    for i in range(np.size(x[:,0])):
        v = __operacao(x[i], w)
        print("para a amostra {}" .format(i) +
              " o neuronio foi: {}" .format(w[v]))


