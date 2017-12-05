import pandas
from pandas import read_csv
import scipy.stats as stats
from sklearn.preprocessing import Imputer
import numpy as np

dataset = read_csv('pimadiabetes.csv', header=None)
dataset[[1,2,3,4,5]] = dataset[[1,2,3,4,5]].replace("0", np.nan)

#dataset.fillna(dataset.mean())

dataset.to_csv('cleanpima.csv',sep=',',encoding='utf-8')


