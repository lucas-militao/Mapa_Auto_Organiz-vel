import numpy as np

from organization.methods import Calculate as cal
from functions import MatrixGenerator as mg

def train(qtdNeuronios, entradas, taxaAprendizagem, raioVizinhanca, precisao):
    N = qtdNeuronios
    x = entradas
    n = taxaAprendizagem
    r = raioVizinhanca
    epoca = 0
    pare = False

    #inicializando a matriz de pesos com valores entre 1 e -1
    w = mg.randomVectorHighLow(1, -1, N, np.size(x[0,:]))
    #inicializa a matriz que irá receber as distâncias euclidianas
    dist = mg.zeroMatrix(np.size(x[:,0]), N)
    #vetor que irá receber o índice que aponta para o neurônio vencedor
    vencedor = mg.zeroVector(np.size(x[:,0]))

    while(epoca < 10000 and pare == False):

        #looping que calcula e armazena todas as distâncias euclidianas
        for i in range(np.size(x[:,0])):
            for j in range(N):
                dist[i,j] = cal.euclideanNorm(x[i], w[j])
        #looping que irá armazenar os índices dos neurônios vencedores
        for i in range(np.size(vencedor)):
            vencedor[i] = np.argmin(dist[i,:]) #O número de vencedores é igual ao número de padrões de entrada. Um vencedor para cada padrão
        #looping que irá calcular e armazenar os pesos com seus respectivos ajustes dos neurônios vencedores
        for i in range(np.size(vencedor)):
            current = int(vencedor[i])
            w[current] = cal.adjustWeights(w[current], n, x[i])



        epoca = 10000
        epoca += 1

    return w
