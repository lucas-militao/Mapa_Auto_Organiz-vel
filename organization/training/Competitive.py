import numpy as np

from organization.methods import Calculate as cal
from functions import Normalization as nm
from functions import MatrixGenerator as mg


def train(entradas, pesos, taxaAprendizagem):
    x = nm.normalization(entradas)
    w = nm.normalization(pesos)

    dist = mg.zeroVector(np.size(pesos[:, 0]))
    menorDistancia = 0

    epoca = 0

    while (epoca < 5):

        for i in range(np.size(x[:, 0])):
            for j in range(np.size(w[:, 0])):
                dist[j] = cal.euclideanNorm(x[i], w[j])

            menorDistancia = np.argmin(dist)

            w[menorDistancia, :] = cal.adjustWeights(w[menorDistancia], taxaAprendizagem, x[i])
            w[menorDistancia, :] = nm.normalization(w[menorDistancia, :])

        epoca += 1

    return w
