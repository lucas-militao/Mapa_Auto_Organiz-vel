from scipy import stats as st
import numpy as np

# def normalization(value):
#     return st.zscore(value)

def normalization(value):
    return (value - np.mean(value))/np.std(value)