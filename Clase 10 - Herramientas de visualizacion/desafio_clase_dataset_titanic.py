# -*- coding: utf-8 -*-
"""
a partir del dataset titanic hacer los siguientes graficos
1 - graficar variables age vs fare y calcular coeficiente de correlacion
2 - calcular un grafico de barras contando la cantidad de gente por 'class'
    dividiendo por la varible 'sex'
"""
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

sns.set_theme(style="darkgrid")

df_titanic = sns.load_dataset('titanic')
# matriz de correlacion
correlacion = df_titanic[['age','fare']].corr()
# coeficiente de correlacion 
coef_correlacion = round(df_titanic[['age','fare']].corr().loc['fare'][0],6)
print(f'Coeficiente de correlacion:{coef_correlacion}')

# 1
plt.figure()
ax = sns.scatterplot(df_titanic['age'],df_titanic['fare'],palette="Set2")
ax.set(xlabel='Edad',ylabel='Tarifa',
       title='Relacion entre Edad y Precio de Tarifa')
plt.savefig('correlacion de edad y tarifa')

# 2
# plt.figure()
ax = sns.countplot(x="class", hue="sex", data=df_titanic,palette="Set2")
ax.set(xlabel='Clase',ylabel='Cantidad',
       title='Cantidad de pasajeros por clase discriminados por sexo')
plt.savefig('cantidad de pasajeros por clase discriminados por sexo')