import numpy as np

from training import  CompetitiveAlgorithm as ca
from scipy import stats as st

def main():

    taxaAprendizagem = 0.3
    x = np.matrix('0.35 0.8')
    w = np.matrix('0.2 0.3;'
                  ' 0.6 0.5; '
                  '0.4 0.7; '
                  '0.9 0.6; '
                  '0.2 0.8')

    print(ca.train(x,w,taxaAprendizagem))





main()