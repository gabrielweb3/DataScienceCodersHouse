# -*- coding: utf-8 -*-
"""
regresiones lineales de datos de anscombe
"""

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


# Import modules for data manipulation, maths, and plotting
import pandas as pd

# Set style of plotting area
import matplotlib.style as style
style.use('seaborn')

raw_anscombe = "https://raw.githubusercontent.com/andrewhetherington/python-projects/master/Blog%E2%80%94Anscombe's%20Quartet/anscombes.csv"
anscombe = pd.read_csv(raw_anscombe)

anscombe.columns = anscombe.columns.str.split('_', expand=True)
anscombe.columns = anscombe.columns.swaplevel(1,0)

print(anscombe)

# # ploteados
# fig = plt.figure(figsize=(12,12))
# # --- FIRST PLOT ---
# # Add top left subplot within plotting area
# ax1 = fig.add_subplot(221)
# # Plot data
# ax1.scatter(anscombe["I"]["x"], anscombe["I"]["y"])
# # Add text
# ax1.text(x=11.5, y = 4.5, s = "",
#             fontsize = 300, alpha = .10, ha="center")
# # Set x-axis limits
# ax1.set_xlim(3,20)
# # Set y-axis limits
# ax1.set_ylim(3,13)
# # --- SECOND PLOT ---
# # Add top right subplot within plotting area
# ax2 = fig.add_subplot(222)
# # Plot data
# ax2.scatter(anscombe["II"]["x"], anscombe["II"]["y"])
# # Add text
# ax2.text(x=11.5, y = 4.5, s = "",
#             fontsize = 300, alpha = .10, ha="center")
# # Set x-axis limits
# ax2.set_xlim(3,20)
# # Set y-axis limits
# ax2.set_ylim(3,13)
# # --- THIRD PLOT ---
# # Add bottom left subplot within plotting area
# ax3 = fig.add_subplot(223)
# # Plot data
# ax3.scatter(anscombe["III"]["x"], anscombe["III"]["y"])
# # Add text
# ax3.text(x=11.5, y = 4.5, s = "",
#             fontsize = 300, alpha = .10, ha="center")
# # Set x-axis limits
# ax3.set_xlim(3,20)
# # Set y-axis limits
# ax3.set_ylim(3,13)
# # --- FOURTH PLOT ---
# # Add bottom left subplot within plotting area
# ax4 = fig.add_subplot(224)
# # Plot data
# ax4.scatter(anscombe["IV"]["x"], anscombe["IV"]["y"])
# # Add text
# ax4.text(x=11.5, y = 4.5, s = "",
#         fontsize = 300, alpha = .10, ha="center")
# # Set x-axis limits
# ax4.set_xlim(3,20)
# # Set y-axis limits
# ax4.set_ylim(3,13)
# # Code for plotting trendlines, if desired
# plt.savefig("anscombe_plotted")


# regresion lineal de II
dfII = anscombe['II']
print(dfII)
# Create linear regression object
regII = LinearRegression()
# Train the model using the training sets
regII.fit(dfII[['x']], dfII['y'])
# Make predictions using the testing set
y_pred =regII.predict(dfII[['x']])
# ploteo de regresion
plt.figure()
plt.scatter(dfII['x'], dfII['y'])
plt.plot(dfII['x'], y_pred,color = 'r')
plt.scatter(dfII['x'], y_pred, color='r')
plt.xlim(0,15)
plt.ylim(0,15)
# coeficientes
print(regII.coef_,regII.intercept_)
R2 = r2_score(dfII['y'], y_pred)
print(R2)
plt.savefig("anscombe_plotted con regresionesII")

# creo set de datos para regresion lineal
dfI = anscombe['I']
# creo objeto para la relacion lineal
regresionI = LinearRegression()
# entreno el modelo usando los x e y de los datos
regresionI.fit(dfI[['x']],dfI['y'])
# realizo prediccion
prediccionY = regresionI.predict(dfI[['x']])
# calculo coeficientes
print(regresionI.coef_,regresionI.intercept_)
R2I = r2_score(dfI['y'], prediccionY)
print(R2I)
# ploteo de regresion
plt.figure()
plt.scatter(dfI['x'], dfI['y'])
plt.plot(dfI['x'], prediccionY,color = 'r')
plt.scatter(dfI['x'], prediccionY, color='r')
plt.xlim(0,15)
plt.ylim(0,15)
plt.savefig("anscombe_plotted con regresionesI")

# creo set de datos para regresion lineal
dfIII = anscombe['III']
# creo objeto para la relacion lineal
regresionIII = LinearRegression()
# entreno el modelo usando los x e y de los datos
regresionIII.fit(dfIII[['x']],dfIII['y'])
# realizo prediccion
prediccionY = regresionIII.predict(dfIII[['x']])
# calculo coeficientes
print(regresionIII.coef_,regresionIII.intercept_)
R2III = r2_score(dfIII['y'], prediccionY)
print(R2III)
# ploteo de regresion
plt.figure()
plt.scatter(dfIII['x'], dfIII['y'])
plt.plot(dfIII['x'], prediccionY,color = 'r',label=R2III)
plt.scatter(dfIII['x'], prediccionY, color='r')
plt.xlim(0,15)
plt.ylim(0,15)
plt.savefig("anscombe_plotted con regresionesIII")

# creo set de datos para regresion lineal
dfIV = anscombe['IV']
# creo objeto para la relacion lineal
regresionIV = LinearRegression()
# entreno el modelo usando los x e y de los datos
regresionIV.fit(dfIV[['x']],dfIV['y'])
# realizo prediccion
prediccionY = regresionIV.predict(dfIV[['x']])
# calculo coeficientes
print(regresionIV.coef_,regresionIV.intercept_)
R2IV = r2_score(dfIV['y'], prediccionY)
print(R2IV)
# ploteo de regresion
plt.figure()
plt.scatter(dfIV['x'], dfIV['y'])
plt.plot(dfIV['x'], prediccionY,color = 'r')
plt.scatter(dfIV['x'], prediccionY, color='r')
plt.xlim(0,15)
plt.ylim(0,15)
plt.savefig("anscombe_plotted con regresionesIV")







