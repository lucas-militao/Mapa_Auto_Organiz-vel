import numpy as np

from functions import TreatData as td
from functions import MatrixGenerator as mg
from organization.training import Kohonen as ko
from organization.methods import Calculate as cal

arquivoX = open("../Xlarge.txt")
data = td.getData(arquivoX)
precisao = pow(10, -4)
taxaAprendizagem = 0.1
numeroNeuronios = 9
raio = 1

def main():
    #exercício dos slides
    # taxaAprendizagem = 0.3
    # x = np.matrix('0.35 0.8')
    w = np.matrix('0.2 0.3;'
                  ' 0.6 0.5; '
                  '0.4 0.7; '
                  '0.1 0.6; '
                  '0.2 0.8')

    #porta xor
    # x = np.matrix('0 0;'
    #               '0 1;'
    #               '1 0;'
    #               '1 1')

    # ko.trainUnidimensional(4, x, taxaAprendizagem, precisao, raio)

    # ko.trainUnidimensional(25, data, taxaAprendizagem, precisao, raio)

    # ko.trainBidimensional(numeroNeuronios, data, taxaAprendizagem, raio, precisao)

    result = cal.distanceFromTwoPoints(1,2,0,1)
    print(result)


main()

#Arquivo Xlarge.txt
# A1 A2
# B1 B2
# C1 C2
# D1 D2
# E1 E2
# J1 J2
# K1 K2