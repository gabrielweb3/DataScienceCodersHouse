# Cargamos las librerías típicas de python
import numpy as np              # numpy para los arrays
import pandas as pd 
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn import preprocessing

# cargo datos
df = pd.read_csv('TelcoCustomerChurn.csv')
# transformo cargos totales a numericos
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'],errors='coerce')

df = df.dropna()

random_seed = 7
# Lista de features que vamos a considerar 
features = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']

# Variable a predecir
target = 'Churn'

# Construcción de la matriz de features
X = df[features].to_numpy()
# Construcción del vector a predecir
y = df[target].to_numpy()

# Creacion de las matrices de entrenamiento y testeo. 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=random_seed)
print('Dimensión de la matriz de features para entrenamiento: {}'.format(X_train.shape))
print('Dimensión de la matriz de features para testeo: {}'.format(X_test.shape))

# Normalizamos en train
scaler_train = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler_train.transform(X_train)

# probar otros números para k
# probar otras distancias, ej: euclidean, minkowski, manhattan 
# probar dar mas peso a los vecinos de un orden superior: weights = 'distance'
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean') 
knn.fit(X_train_scaled, y_train)

y_pred_train = knn.predict(X_train_scaled)
accuracy_train =  accuracy_score(y_pred_train, y_train)
print('El accuracy en el conjunto de train es', accuracy_train)

# Normalizamos en test
scaler_test = preprocessing.StandardScaler().fit(X_test)
X_test_scaled = scaler_test.transform(X_test)

y_pred_test = knn.predict(X_test_scaled)
accuracy_test =  accuracy_score(y_pred_test, y_test)
print('El accuracy en el conjunto de test es', accuracy_test)

plot_confusion_matrix(knn, X_test_scaled, y_test)  
plt.show()


"""
Ventajas de KNN:
1.Fácil de usar e interpretar (podemos entender por qué obtuvimos la predicción que obtuvimos)

2.Depende de un único hiperparámetro

4.Entrenamiento súper rápido

5.Útil para sistemas de recomendaciones

6."Buena" performance

Desventajas de KNN:
1.Es lo que se llama un “ lazy learner ”: no se estiman los parámetros de una f(x,β) que pueda ser aplicada rápidamente a nuevos datos; cada nueva predicción necesita potencialmente todos los datos.

2.Por el ítem anterior, se vuelve lento a la hora de predecir muchos labels

3.Puede tener requerimientos altos de memoria
"""