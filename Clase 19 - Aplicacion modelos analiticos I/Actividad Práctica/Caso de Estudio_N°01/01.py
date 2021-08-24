# Cargamos las librerías típicas de python
# importamos las librerías usuales de python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importamos los algoritmos de clusterizacion a utilizar en esta notebook
from sklearn.cluster import KMeans                    # K-means
from sklearn.cluster import AgglomerativeClustering   # Clustering jerárquicofrom sklearn import preprocessing
from sklearn import preprocessing

# Estandarizador (transforma las variables en z-scores)
from sklearn.preprocessing import StandardScaler
std_scale = StandardScaler() # Creamos el estandarizador

# carga de datos
df = pd.read_csv('BankMarketing.csv',sep=';')


# from sklearn.datasets import load_iris
# data = load_iris()
species = [data['target_names'][x] for x in data['target']]
df = pd.DataFrame(np.column_stack([data['data']]), columns=data['feature_names'])


# # cambiar titulos columnas
# for i in range(0,len(df.columns)):
#     df[i].rename(str(df[i][0]))
#     # print(df[i][0])
    
# print(df.columns)

# configuracion de modelo
random_seed = 7
# Lista de features que vamos a considerar 
features = [df['age'], df['marital'], df['education'], df['month']]

# Variable a predecir
target = df['poutcome']

# Construcción de la matriz de features
X = df[features].to_numpy()
# Construcción del vector a predecir
y = df[target].to_numpy()

# Creacion de las matrices de entrenamiento y testeo. 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=random_seed)
print('Dimensión de la matriz de features para entrenamiento: {}'.format(X_train.shape))
print('Dimensión de la matriz de features para testeo: {}'.format(X_test.shape))