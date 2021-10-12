# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 11:20:56 2021

@author: gabri
"""

import numpy as np              # numpy para los arrays
import pandas as pd 
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn import preprocessing
import matplotlib.pyplot as plt
import matplotlib as mpl

df = pd.read_csv('C:/Users/gabri/OneDrive/Escritorio/Cursos/Data Science/DataScienceCodersHouse/Clase 22 - Algoritmos y validacion de modelo de ML/diabetes.csv')

random_seed = 7
# Lista de features que vamos a considerar 
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

# Variable a predecir
target = 'Outcome'

# Construcción de la matriz de features
X = df[features].to_numpy()
# print('Matriz de entradas ',X)
# Construcción del vector a predecir
y = df[target].to_numpy()
# print('Vector a predecir: ',y)

# Creacion de las matrices de entrenamiento y testeo. 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=random_seed)
print('Dimensión de la matriz de features para entrenamiento: {}'.format(X_train.shape))
print('Dimensión de la matriz de features para testeo: {}'.format(X_test.shape))

# Normalizamos en train
scaler_train = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler_train.transform(X_train)
# normalizo el test
X_test_scaled = scaler_train.transform(X_test)

# probar otros números para k
# probar otras distancias, ej: euclidean, minkowski, manhattan 
# probar dar mas peso a los vecinos de un orden superior: weights = 'distance'
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean') 
knn.fit(X_train_scaled, y_train)

y_pred_train = knn.predict(X_train_scaled)
accuracy_train =  accuracy_score(y_pred_train, y_train)
print('El accuracy en el conjunto de train es', accuracy_train)

plot_confusion_matrix(knn, X_test_scaled, y_test)  
plt.show()

VP=81;VN=33;FP=16;FN=24
# accuracy es lo cerca que esta el resultado de una medicion del valor verdadero
print('Exactitud:',(VP+VN)/(VP+VN+FN+VN)*100) 
# dispersion del conjunto de valores obtenidos a partir de mediciones repetidas de una magnitud
print('Precision:',VP/(VP+FP))
# sensibilidad: proporcion de casos positivios que fueron correctamente identificados
print('Sensibilidad:',VP/(VP+FN))
# especifidad, casos negativos que se detectaron correctamente
print('Especifidad:',VN/(VN+FP))