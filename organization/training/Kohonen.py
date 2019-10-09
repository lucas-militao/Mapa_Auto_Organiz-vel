import numpy as np

from organization.methods import Calculate as cal
from functions import MatrixGenerator as mg

def __operacao(x, w):

    dist = mg.zeroVector(np.size(w[:,0]))

    for i in range(np.size(dist)):
        dist[i] = cal.euclideanNorm(x, w[i])

    vencedor = np.argmin(dist)

    return vencedor

def __unidimensional(qtdNeuronios, entradas, pesos, taxaAprendizagem, precisao):
    N = qtdNeuronios
    x = entradas
    w = pesos
    n = taxaAprendizagem
    epoca = 0
    pare = False

    # inicializa o vetor que irá receber as distâncias euclidianas
    dist = mg.zeroVector(N)

    while(epoca < 10000 and pare != True):

        #calcula as distancias euclidianas
        for i in range(np.size(dist)):
            dist[i] = cal.euclideanNorm(x, w[i])
        #definir vencedor
        vencedor = np.argmin(dist)
        #ajustar pesos do vencedor
        w[vencedor] = cal.adjustWeightsWinner(w[vencedor], n, x)
        #verificar se houve mudança significativa
        pare = cal.checkChange(x, w[vencedor], precisao)

        epoca += 1

    return w



def trainUnidimensional (qtdNeuronios, entradas, taxaAprendizagem, raioVizinhanca, precisao):
    N = qtdNeuronios
    x = entradas
    n = taxaAprendizagem
    r = raioVizinhanca
    epoca = 0
    pare = False

    #inicializando a matriz de pesos com valores entre 1 e -1
    w = mg.randomVectorHighLow(1, -1, N, np.size(x[0,:]))

    #treinamento
    for i in range(np.size(x[:,0])):
        w = __unidimensional(N, x[i], w, n, precisao)

    for i in range(np.size(x[:,0])):
        v = __operacao(x[i], w)
        print("cluster: {}" .format(x[i]) +
              "\npeso: {}" .format(w[v]))
