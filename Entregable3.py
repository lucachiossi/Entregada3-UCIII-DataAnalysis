# Import Libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy as sci

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

folder = "C:/Users/lucac_000/Desktop/Luca/UNIVERSITA/__MAGISTRALE__/_II_anno/M1_Analisis_de_Datos/___Lboratory/documents_lab_3/"

# Read DataSet
data_set = pd.read_csv(folder + "meteo_calidad_2015.csv", decimal=",", sep=";")

f = open(folder + "dataset.txt", "w")
f.write(data_set.to_string())
f.close()

# Delete "Dia" column
data_set.drop('Dia', axis=1, inplace=True)

f = open(folder + "dataset_ready.txt", "w")
f.write(data_set.to_string())
f.close()

# Read data description
f = open(folder + "dataset_describe.txt","w")
f.write(data_set.describe().to_string())
f.close()

# Calculate the max temperature of Jan
data_set_January = data_set[data_set.Mes == 'ENE']
f = open(folder + "dataset_only_jan.txt","w")
f.write(data_set_January.to_string())
f.close()
f = open(folder + "dataset_only_jan_describe.txt","w")
f.write(data_set_January.describe().to_string())
f.close()

# Calculate the max temperature of each month
data_set_month = data_set.groupby(['Mes'])
f = open(folder + "dataset_by_month.txt","w")
f.write(data_set_month.describe().to_string())
f.close()

# Similitude rate between max-temperature and normal-distribution
z = (data_set['T_MAX']-np.mean(data_set['T_MAX']))/np.std(data_set['T_MAX'])
sci.stats.probplot(z, dist="norm", plot=plt)
plt.title("Normal 0-0 plot")
plt.show()

# Histogram
# attr = data_set["T_MAX"]
# plt.hist(attr, bins=25)
# plt.show()
