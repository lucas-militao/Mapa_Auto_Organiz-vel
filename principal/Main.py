import numpy as np

from functions import TreatData as td
from organization.training import Kohonen as ko

arquivoX = open("../Xlarge.txt")
data = td.getData(arquivoX)
precisao = pow(10, -4)
numeroNeuronios = 9
raio = 0

def main():

    taxaAprendizagem = 0.3
    x = np.matrix('0.35 0.8')
    w = np.matrix('0.2 0.3;'
                  ' 0.6 0.5; '
                  '0.4 0.7; '
                  '0.9 0.6; '
                  '0.2 0.8')

    print(ko.train(numeroNeuronios, data, precisao, raio, precisao))

    # print(ca.train(x, w, taxaAprendizagem))

    # print(tdata.getData(arquivoX))


main()