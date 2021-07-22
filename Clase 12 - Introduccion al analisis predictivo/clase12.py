# pca video 
# https://www.youtube.com/watch?v=FgakZw6K1QQ
# pca
# definir un sistema de coordenanas uevos de tal forma que maximize
# la varianza de la distribucion

"""
Punto de partida
planteamos hipotessis de quenpodria existir algun tipo de 
dependencia entre las varaibles
si esta dependencia existe, queremos ver de que forma se da 
esta relacion
variables con correlacion positiva

si tenemos un conjunto de puntos en las variables x e y, 
y de alguna forma que depende de x, una forma es trazar una recta
que de alguna manera puede representar a esos puntos,
tomando un criterio para la representacion y trazar una 
recta que cumpla con el
luego se hace un ajuste de la recta a los datos
se realiza ajuste por metodo de minimos cuadrados
este metodo minimiza el y-f(x)
sum[y-(a+xb)]^2

el coeficiente de correlacion pearson al cuadrado
R^2 - representa el pocentaje de variablidad de los datos,
explicada por el modelo de regresion lineal

"""

# REGRESION LINEAL


import matplotlib as mpl
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
import numpy as np
mpl.style.use('bmh')
import seaborn as sns
from sklearn import datasets


import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

# Load the diabetes dataset
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

# Use only one feature
diabetes_X = diabetes_X[:, np.newaxis, 2]

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

# cantidad de lineas de test de x e y deben ser iguales

# Create linear regression object
regr = LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

# Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(diabetes_y_test, diabetes_y_pred))

regr.coef_

# Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')

plt.scatter(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)
plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)



















