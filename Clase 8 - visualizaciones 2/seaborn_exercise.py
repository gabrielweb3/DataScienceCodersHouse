# -*- coding: utf-8 -*-
"""
clase 8 visualizaciones 2
SEABORN
"""

# import pandas as pd
import matplotlib.pyplot as plt
# import matplotlib as mpl
import seaborn as sns
sns.set()
# cargo, limpio y muestro data
df_ejercicio = sns.load_dataset('exercise')
df_ejercicio = df_ejercicio.drop('Unnamed: 0',axis='columns')
df_ejercicio.head()

# se extraen observaciones que corresponden con un ejercicio 
# de 30 minutos
df_30min = df_ejercicio[df_ejercicio['time']=='30 min']
df_30min.head()

# se grafican distribuciones con sns.displot
plt.figure()
ax = sns.displot(data=df_30min, x='pulse',
                 kind='kde', hue='kind',fill=True)
ax.set(xlabel='Frecuencia Cardiaca',ylabel='Densidad',
       title='Distribucion de pulsaciones')
ax.savefig('distribucion pulsaciones sns.pdf')

# distribucion en base a dietas
plt.figure()
ax = sns.displot(data=df_30min, x='pulse',
                 kind='kde', hue='diet',fill=True)
ax.set(xlabel='Frecuencia Cardiaca',ylabel='Densidad',
       title='Distribucion de pulsaciones')
ax.savefig('distribucion dietas sns.pdf')

# histograma de pulsaciones dependiendo ejercicio
plt.figure()
ax = sns.displot(data=df_30min, x='pulse',
                  kind='hist', hue='kind',fill=True)
ax.set(xlabel='Frecuencia Cardiaca',ylabel='Densidad',
        title='Distribucion de pulsaciones')
ax.savefig('historial pulsaciones sns.pdf')

# categorical plots
ax = sns.catplot(data=df_ejercicio, kind='violin', 
                 x='time', y='pulse', hue='diet', 
                 split=True)
ax.set(xlabel='Duración de ejercicio', 
       ylabel='Frecuencia cardíaca', 
       title='Categorización de la distribución de pulsaciones')
ax.savefig('categorizacion pulsaciones sns.pdf')

ax = sns.catplot(data=df_ejercicio, kind='violin',
                 x='kind', y='pulse',
                 hue='diet', split=True)
ax.set(xlabel='Duración de ejercicio',
       ylabel='Frecuencia cardíaca',
       title='Categorización de la distribución de pulsaciones')
ax.savefig('categorizacion en base a actividad fisica y dieta sns.pdf')