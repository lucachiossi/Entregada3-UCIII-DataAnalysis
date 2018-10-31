## Import Libraries

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

# folder = "C:/Users/lowei/Desktop/Espagne/Cours/Data Analysis/Practica 2/"
folder = "C:/Users/lucac_000/Desktop/Luca/UNIVERSITA/__MAGISTRALE__/_II_anno/M1_Analisis_de_Datos/___Lboratory/documents_lab_3/shared work/"


## Read DataSet
data_set = pd.read_csv(folder + "meteo_calidad_2015.csv", decimal=",", sep=";")

# Import dataset and renaming columns
data_set.rename(columns ={'TOL':'Tolueno'}, inplace=True)
data_set.rename(columns ={'TOL_MAX':'Tolueno_MAX'}, inplace=True)
data_set.rename(columns ={'BEN':'Benceno'}, inplace=True)
data_set.rename(columns ={'BEN_MAX':'Benceno_MAX'}, inplace=True)
data_set.rename(columns ={'EBE':'Ethilbenceno'}, inplace=True)
data_set.rename(columns ={'EBE_MAX':'Ethilbenceno_MAX'}, inplace=True)
data_set.rename(columns ={'MXY':'Metaxileno'}, inplace=True)
data_set.rename(columns ={'MXY_MAX':'Metaxileno_MAX'}, inplace=True)
data_set.rename(columns ={'PXY':'Paraxileno'}, inplace=True)
data_set.rename(columns ={'PXY_MAX':'Paraxileno_MAX'}, inplace=True)
data_set.rename(columns ={'OXY':'Ortoxileno'}, inplace=True)
data_set.rename(columns ={'OXY_MAX':'Ortoxileno_MAX'}, inplace=True)
data_set.rename(columns ={'TCH':'Hidrocarburos totales'}, inplace=True)
data_set.rename(columns ={'TCH_MAX':'Hidrocarburos totales_MAX'}, inplace=True)
data_set.rename(columns ={'NMCH':'Hidrocarburos no metánicos'}, inplace=True)
data_set.rename(columns ={'NMCH_MAX':'Hidrocarburos no metánicos_MAX'}, inplace=True)

# Delete "Dia" column
data_set.drop('Dia', axis=1, inplace=True)

f = open(folder + "dataset.txt", "w")
f.write(data_set.to_string())
f.close()

## Read data description
f = open(folder + "dataset_describe.txt","w")
f.write(data_set.describe().to_string())
f.close()

## Calculate the max temperature of Jan
data_set_January = data_set[data_set.Mes == 'ENE']
f = open(folder + "dataset_only_jan.txt","w")
f.write(data_set_January.to_string())
f.close()
f = open(folder + "dataset_only_jan_describe.txt","w")
f.write(data_set_January.describe().to_string())
f.close()

## Calculate the max temperature of each month
data_set_month = data_set.groupby(['Mes'])
f = open(folder + "dataset_by_month.txt","w")
f.write(data_set_month.describe().to_string())
f.close()
print (data_set_month['T_MAX'].agg(np.mean))
print("Temperatura maximal media del ano" , data_set_month['T_MAX'].agg(np.mean).agg(np.mean), "°C")

## QQ plot: Similitude rate between max-temperature and normal-distribution
z = (data_set['T_MAX']-np.mean(data_set['T_MAX']))/np.std(data_set['T_MAX'])
sci.stats.probplot(z, dist="norm", plot=plt)
plt.title("Normal Q-Q plot")
plt.show()

# Histogram
plt.hist(z, bins=50, density=1)
plt.title("Histograma")
plt.show()

# Kurtosis
sci.stats.kurtosis(z, bias=False) # No funciona
plt.title("kurtosis plot")
plt.show()

## Graph T MAX / CO & O3
df= data_set.sort_values(['T_MAX','CO'],ascending=True)
plt.plot(df['T_MAX'], df['CO'])
plt.title("La concentración de CO frente a la temperatura máxima")
plt.show()


df= data_set.sort_values(['T_MAX','O3'],ascending=True)
plt.plot(df['T_MAX'], df['O3'])
plt.title("La concentración de Ozono frente a la temperatura máxima")
plt.show()

## Pairplot
sns.jointplot(data_set['T_MAX'],data_set['CO'], kind="reg")
plt.show()
plt.close()
sns.jointplot(data_set['T_MAX'],data_set['O3'], kind="reg")
plt.show()
plt.close()

## Correlation Matrix
data_set_corr = data_set
data_set_corr['Mes'] = data_set_corr['Mes'].map({'ENE': 1, 'FEB': 2, 'MAR': 3, 'ABR': 4,
                                                 'MAY': 5, 'JUN': 6, 'JUL': 7, 'AGO': 8,
                                                 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DIC': 12})
f = open(folder + "dataset_corr.txt","w")
f.write(data_set_corr.corr().to_string())
f.close()
NO2 = data_set_corr.corr()['NO2']
print(NO2.sort_values())

## HeatMap
mask = np.zeros((31,31))
for i in range (0,31):
    for j in range (0,31):
        if i<j:
            mask[i,j]=True
sns.heatmap(data_set_corr.corr(), vmin=-1, vmax=+1, cmap="bwr", center=0, linewidths=0.4, cbar= True, square = True, mask = mask)
plt.show()
plt.close()


## Delete columns                       
data_set.drop('SO2_MAX', axis=1, inplace=True)
data_set.drop('CO_MAX', axis=1, inplace=True)
data_set.drop('NO_MAX', axis=1, inplace=True)
data_set.drop('NO2_MAX', axis=1, inplace=True)
data_set.drop('PM2.5_MAX', axis=1, inplace=True)
data_set.drop('PM10_MAX', axis=1, inplace=True)
data_set.drop('O3_MAX', axis=1, inplace=True)
data_set.drop('Tolueno_MAX', axis=1, inplace=True)
data_set.drop('Benceno_MAX', axis=1, inplace=True)
data_set.drop('Ethilbenceno_MAX', axis=1, inplace=True)
data_set.drop('Hidrocarburos totales_MAX', axis=1, inplace=True)
data_set.drop('Hidrocarburos no metánicos_MAX', axis=1, inplace=True)

f = open(folder + "dataset_V2.txt", "w")
f.write(data_set.to_string())
f.close()

## NO2 con Viento_Max

NO2 = data_set['NO2']
viento_max = data_set['Viento_MAX']
T_max = data_set['T_MAX']
lluvia = data_set['Lluvia']
regressor = LinearRegression()

regressor.fit(viento_max, NO2)
print("R cuadrado test = ", regressor.score(NO2, viento_max))

## NO2 con Viento_MAX y T_MAX

X = data_set[['Viento_MAX', 'T_MAX']]
for train_index, test_index in split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = NO2.iloc[train_index], NO2.iloc[test_index]
    regressor.fit(X_train, y_train)
    print("R cuadrado test = ", regressor.score(X_test, y_test))
    y_pred = regressor.predict(X_test)