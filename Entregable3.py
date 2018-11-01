## Import Libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy as sci
import statsmodels.api as sm

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.tree import DecisionTreeRegressor

# As we were sending the code to each other using GitKraken, it was easier to just comment and uncomment the wrong and right path
folder = "C:/Users/lowei/Desktop/Espagne/Cours/Data Analysis/Practica 2/"
#folder = "C:/Users/lucac_000/Desktop/Luca/UNIVERSITA/__MAGISTRALE__/_II_anno/M1_Analisis_de_Datos/___Lboratory/documents_lab_3/shared work/"


## Read DataSet
data_set = pd.read_csv(folder + "meteo_calidad_2015.csv", decimal=",", sep=";")

# Renaming columns to be more understandable
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

# Delete "Dia" column because it is irrelevant
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
print (data_set_month['T_MAX'].agg(np.mean)) # show the average temperature of each month
print("Temperatura maximal media del ano" , data_set_month['T_MAX'].agg(np.mean).agg(np.mean), "°C") # Average temperature during the year

## QQ plot: Similitude rate between max-temperature and normal-distribution
z = (data_set['T_MAX']-np.mean(data_set['T_MAX']))/np.std(data_set['T_MAX'])
sci.stats.probplot(z, dist="norm", plot=plt)
plt.title("Normal Q-Q plot")
plt.show()

# Histogram
plt.hist(z, bins=50, density=1)
plt.title("Histograma")
plt.show()

# Skew
print("Skew:")
print(data_set['T_MAX'].skew())

# Kurtosis
print("Kurtosis:")
print(data_set['T_MAX'].kurtosis())

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
data_set_corr['Mes'] = data_set_corr['Mes'].map({'ENE': 1, 'FEB': 2, 'MAR': 3,'ABR': 4, 'MAY': 5, 'JUN': 6, 'JUL': 7, 'AGO': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DIC': 12})
data_set_corr['Dia_sem']= data_set_corr['Dia_sem'].map({'L':1,'M':2,'X':3,'J':4,'V':5,'S':6,'D':7}) #changing nominal attributes to numeric in order to see their influence in the correlation matrix
f = open(folder + "dataset_corr.txt","w")
f.write(data_set_corr.corr().to_string())
f.close()
NO2_corr = data_set_corr.corr()['NO2']
print(NO2_corr.sort_values()) # Print all the correlation coeff to see which one is the most related to NO2

## HeatMap
mask = np.zeros((32,32))
for i in range (0,32):
    for j in range (0,32):
        if i<j:
            mask[i,j]=True
sns.heatmap(data_set_corr.corr(), vmin=-1, vmax=+1, cmap="bwr", center=0, linewidths=0.4, cbar= True, square = True, mask = mask)
plt.show()
plt.close()
# We can see 2 diagonals. The main one which is an attribute in front of himself. The second one is an attribute in front of himself_max. Therefore, it is not relevante to keep both attributes since they give the same informations.

## Delete columns                       
data_set.drop('Dia_sem', axis=1, inplace=True) #Low correlation coeff
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

## HeatMap V2
# Showing the heat map without all the useless attributes
mask = np.zeros((19,19))
for i in range (0,19):
    for j in range (0,19):
        if i<j:
            mask[i,j]=True
sns.heatmap(data_set.corr(), vmin=-1, vmax=+1, cmap="bwr", center=0, linewidths=0.4, cbar= True, square = True, mask = mask)
plt.show()
plt.close()

## NO2 con Viento_Max

# print(data_set['NO2'].shape)
# print(data_set['Viento_MAX'].shape)
NO2 = data_set['NO2'].values
viento_max = data_set['Viento_MAX'].values.reshape(-1,1)
regr = LinearRegression()
regr.fit(viento_max,NO2)
print("R cuadrado = ", regr.score(viento_max,NO2))
print("coeff lineal = ",regr.coef_)
print("intersection punto = ",regr.intercept_)
print("NO2 = ", regr.coef_[0], "Viento_MAX + ", regr.intercept_)
print (np.polyfit(data_set['Viento_MAX'],data_set['NO2'],1)) # confirmation with another function

## NO2 con Viento MAx y T_MAX
Viento_y_T_max = data_set[['Viento_MAX','T_MAX']]
regr.fit(Viento_y_T_max,NO2)
print("R cuadrado = ", regr.score(Viento_y_T_max,NO2))
print("coeff lineal = ",regr.coef_)
print("intersection punto = ",regr.intercept_)
print("NO2 = ", regr.coef_[0], "Viento_MAX + ", regr.coef_[1], "T_MAX + ", regr.intercept_)

## NO2 con Viento_Max, T_Max y Lluvia
Viento_T_lluvia = data_set[['Lluvia','T_MAX','Viento_MAX']]
regr.fit(Viento_T_lluvia,NO2)
print("R cuadrado = ", regr.score(Viento_T_lluvia,NO2))
print("coeff lineal = ",regr.coef_)
print("intersection punto = ",regr.intercept_)
print("NO2 = ", regr.coef_[2], "Viento_MAX + ", regr.coef_[1], "T_MAX + ", regr.coef_[0],"Lluvia + ", regr.intercept_)

## Multicolinealidad
vif = [variance_inflation_factor(data_set.values,i) for i in range(data_set.shape[1])]
for i in range(len(vif)):
        print(vif[i]) # V.I.F. = 1/(1-R^2)

## Polinómico para explicar NO2
for i in range(1,5):
        poly = np.polyfit(data_set['NO2'],data_set[['Viento_MAX','T_MAX','Lluvia']],i)
        print(poly)
# the equations only link one attribute with NO2. We did not manage to find the equation linking NO2 with all the three attributes at the same time
     
## Regresión múltiple no lineal basado en árboles
multArbol = DecisionTreeRegressor()
multArbol.fit(Viento_T_lluvia, NO2)
yMultArbolPred = multArbol.predict(Viento_T_lluvia)
plt.scatter(x=NO2, y=yMultArbolPred)
print("R cuadrado = ", multArbol.score(Viento_T_lluvia, NO2))


## Create a Class column
data_set['target']= pd.Series(np.zeros(len(data_set['NO2'])), index=data_set.index) # Spliting the datas in 4 folds to keep relevant size of fold
for i in range (len(data_set['NO2'])):
    if data_set['NO2'][i] < 35:
        data_set['target'][i]=0
    elif data_set['NO2'][i] < 60:
        data_set['target'][i]=1
    elif data_set['NO2'][i]< 85:
        data_set['target'][i]=2
    else :
        data_set['target'][i]=3


## Regresor Logistico / Finding test size 
list=[0.1, 0.2, 0.3, 0.4, 0.5 ,0.6, 0.7, 0.8,0.9]
entretamientoLO=[]
testLO = []
for i in list:  
    y = data_set.target
    X = data_set.drop('target', axis = 1)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = i, random_state = 0)
    # Ajustando/entrenando el modelo
    modelo = LogisticRegression()
    modelo.fit(X_train, y_train)
    # Mostrar R cuadrado
    entretamientoLO.append(modelo.score(X_train, y_train))
    # Prediction en test
    y_pred = modelo.predict(X_test)
    # Mostrar R cuadrado en test
    testLO.append(modelo.score(X_test,y_test))
print("R cuadrado (entretamientoLO): ",entretamientoLO)
print( "R cuadrado (testLO): ",testLO)
plt.plot(list, entretamientoLO,'b', label='Entretamiento')
plt.plot(list, testLO,'r',label='Test')
plt.legend()
plt.xlabel('Percentage of testing data')
plt.ylabel('R²')
plt.title('Regresion LOGISTICO')
plt.show()
plt.clf()
# The best result is obtained for test_size = 0.5


## Regresor Logistico with T_Max, Lluvia and Viento_Max
# This is to see how exact the prediction could have been with only these attributes. Results are very low and show that they are not suffisant. The other attributes are also relevant 
list=[0.1, 0.2, 0.3, 0.4, 0.5 ,0.6, 0.7, 0.8,0.9]
entretamientoLO=[]
testLO = []
for i in list:  
    y = data_set.target
    X = data_set[['T_MAX','Viento_MAX','Lluvia','target']].drop('target', axis = 1)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = i, random_state = 0)
# Ajustando/entrenando el modelo
    modelo = LogisticRegression()
    modelo.fit(X_train, y_train)
# Mostrar R cuadrado
    entretamientoLO.append(modelo.score(X_train, y_train))
# Prediction en test
    y_pred = modelo.predict(X_test)
# Mostrar R cuadrado en test
    testLO.append(modelo.score(X_test,y_test))
print("R cuadrado (entretamientoLO): ",entretamientoLO)
print( "R cuadrado (testLO): ",testLO)
plt.plot(list, entretamientoLO,'b', label='Entretamiento')
plt.plot(list, testLO,'r',label='Test')
plt.legend()
plt.xlabel('Percentage of testing data')
plt.ylabel('R²')
plt.title('Regresion LOGISTICO')
plt.show()
plt.clf()

## REGRESSOR LOGISTICO / Prediction
cpt0=0; cpt1=0; cpt2=0; cpt3=0
y = data_set.target
X = data_set.drop('target', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.5, random_state = 0)
# Ajustando/entrenando el modelo
modelo = LogisticRegression()
modelo.fit(X_train, y_train)
# Mostrar R cuadrado
print('R cuadrado (entrenamiento)= ', modelo.score(X_train, y_train))
# Prediction en test
y_pred = modelo.predict(X_test)
# Mostrar R cuadrado en test
#print(y_pred-y_test)
for j in range (1,len(y_pred)-2):
    if y_pred[j]-np.array(y_test)[j] == 0:
        cpt0 +=1
    elif abs(y_pred[j] - np.array(y_test)[j]) == 1:
        cpt1 +=1
    elif abs(y_pred[j] - np.array(y_test)[j]) == 2:
        cpt2 +=1
    elif abs(y_pred[j] -np.array( y_test)[j]) == 3:
        cpt3 +=1
# this shows how many values were correctly predicted (cpt0), how many are not in the good class but in the closest one (cpt1), how many are two class away (cpt2) ...
print ('cpt0: ', cpt0); print ('cpt1: ', cpt1); print ('cpt2: ', cpt2); print ('cpt3: ', cpt3)
print('R cuadrado(test)= ', modelo.score(X_test, y_test))
print('R cuadrado(pred)= ', modelo.score(X_test, y_pred))

## MODELO NAIVE-BAYESIANO / Finding test size
list=[0.1, 0.2, 0.3, 0.4, 0.5 ,0.6, 0.7, 0.8,0.9]
entretamientoBN=[]
testBN = []
modelo = GaussianNB()
for i in list:  
    y = data_set.target
    X = data_set.drop('target', axis = 1)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = i, random_state = 0)
    # Ajustando/entrenando el modelo
    modelo.fit(X_train, y_train)
    # Mostrar R cuadrado
    entretamientoBN.append(modelo.score(X_train, y_train))
    # Prediction en test
    y_pred = modelo.predict(X_test)
    # Mostrar R cuadrado en test
    testBN.append(modelo.score(X_test,y_test))
print("R cuadrado (entretamientoBN): ",entretamientoBN)
print("R cuadrado (testBN): ",testBN)

plt.plot(list, entretamientoBN,'b', label='Entretamiento')
plt.plot(list, testBN,'r',label='Test')
plt.legend()
plt.xlabel('Percentage of testing data')
plt.ylabel('R²')
plt.title('Regresion BN')
plt.show()
plt.clf()

## MODELO NAIVE-BAYESIANO / Prediction
cpt0=0; cpt1=0; cpt2=0; cpt3=0
y = data_set.target
X = data_set.drop('target', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.1, random_state = 0)
# Ajustando/entrenando el modelo
modelo = GaussianNB()
modelo.fit(X_train, y_train)
# Mostrar R cuadrado
print('R cuadrado (entrenamiento)= ', modelo.score(X_train, y_train))
# Prediction en test
y_pred = modelo.predict(X_test)
# Mostrar R cuadrado en test
# print(y_pred-y_test)
print('R cuadrado(test)= ', modelo.score(X_test, y_test))
print('R cuadrado(pred)= ', modelo.score(X_test, y_pred))
for j in range (1,len(y_pred)-2):
    if y_pred[j]-np.array(y_test)[j] == 0:
        cpt0 +=1
    elif abs(y_pred[j] - np.array(y_test)[j]) == 1:
        cpt1 +=1
    elif abs(y_pred[j] - np.array(y_test)[j]) == 2:
        cpt2 +=1
    elif abs(y_pred[j] -np.array( y_test)[j]) == 3:
        cpt3 +=1

print ('cpt0: ', cpt0); print ('cpt1: ', cpt1); print ('cpt2: ', cpt2); print ('cpt3: ', cpt3)

## Tree Regressor - training
list=[0.1, 0.2, 0.3, 0.4, 0.5 ,0.6, 0.7, 0.8,0.9]
entretamientoTree=[]
testTree= []
multArbol = DecisionTreeRegressor()

for i in list:  
    y = data_set.target
    X = data_set.drop('target', axis = 1)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = i, random_state = 0)
    multArbol.fit(X_train, y_train)
    entretamientoTree.append( multArbol.score(X_train, y_train))
    y_pred =  multArbol.predict(X_test)
    testTree.append(multArbol.score(X_test,y_test))
print("R cuadrado (entretamientoTree): ",entretamientoTree)
print("R cuadrado (testTree): ",testTree)

plt.plot(list, entretamientoTree,'b', label='Entretamiento')
plt.plot(list, testTree,'r',label='Test')
plt.legend()
plt.xlabel('Percentage of testing data')
plt.ylabel('R²')
plt.title('Tree Regressor')
plt.show()
plt.clf()

## Tree Regressor - Prediction
cpt0=0; cpt1=0; cpt2=0; cpt3=0
y = data_set.target
X = data_set.drop('target', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.4, random_state = 0) # test_size=0.1 was better but the test is more relevant on more data. Test_size = 0.4 is a good compromise to have a good prediction and a test on a large amount of data
multArbol = DecisionTreeRegressor()
multArbol.fit(X_train, y_train)
# Mostrar R cuadrado
print('R cuadrado (entrenamiento)= ',multArbol.score(X_train, y_train))
# Prediction en test
y_pred = multArbol.predict(X_test)
# Mostrar R cuadrado en test
# print(y_pred-y_test)
print('R cuadrado(test)= ', multArbol.score(X_test, y_test))
print('R cuadrado(pred)= ', multArbol.score(X_test, y_pred))
for j in range (1,len(y_pred)-2):
    if y_pred[j]-np.array(y_test)[j] == 0:
        cpt0 +=1
    elif abs(y_pred[j] - np.array(y_test)[j]) == 1:
        cpt1 +=1
    elif abs(y_pred[j] - np.array(y_test)[j]) == 2:
        cpt2 +=1
    elif abs(y_pred[j] -np.array( y_test)[j]) == 3:
        cpt3 +=1

print ('cpt0: ', cpt0); print ('cpt1: ', cpt1); print ('cpt2: ', cpt2); print ('cpt3: ', cpt3)