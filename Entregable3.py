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

# Correlation Matrix
data_set_corr = data_set.corr()
f = open(folder + "dataset_corr.txt","w")
f.write(data_set_corr.to_string())
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
plt.title("Normal Q-Q plot")
plt.show()

# Histogram
plt.hist(z, bins=10, density=1)
plt.title("Histograma")
plt.show()


# In[9]:


sci.stats.kurtosis(z, bias=False) # No funciona
plt.title("kurtosis plot")
plt.show()


# In[16]:


df= data_set.sort_values(['T_MAX','CO'],ascending=True)
plt.plot(df['T_MAX'], df['CO'])
plt.title("La concentración de CO frente a la temperatura máxima")
plt.show()


# In[17]:


df= data_set.sort_values(['T_MAX','O3'],ascending=True)
plt.plot(df['T_MAX'], df['O3'])
plt.title("La concentración de Ozono frente a la temperatura máxima")
plt.show()


# In[15]:


sns.jointplot(data_set['T_MAX'],data_set['CO'], kind="reg")


# In[16]:


sns.jointplot(data_set['T_MAX'],data_set['O3'], kind="reg")


# In[18]:


sns.pairplot(data_set['T_MAX'],data_set['O3'])
