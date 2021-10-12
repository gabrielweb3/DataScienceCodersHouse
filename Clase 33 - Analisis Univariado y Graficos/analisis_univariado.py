# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 20:45:25 2021

@author: gabri
"""

import pandas as pd
# import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('Iris.csv')

# descripcion de los datos
estadisticas = df.describe()
print(estadisticas)

# agrupo por species
agrupacion_por_especies = df.groupby(['Species']).mean()
agrupacion_por_especies = agrupacion_por_especies.drop(['Id'],axis=1)
print(agrupacion_por_especies)

# scatter normal
sns.scatterplot(x=df['SepalLengthCm'],
                y=df['SepalWidthCm'],
                hue=df['Species'])

# scatter mas prolijo
sns.FacetGrid(df,hue='Species',size=5)\
.map(plt.scatter,'SepalLengthCm','SepalWidthCm')\
.add_legend()

# comparacion con distribucion
sns.jointplot(x='SepalLengthCm',
              y='SepalWidthCm',
              data=df,size=5)

# scatter ordenado
sns.swarmplot(x=df['Species'],y=df['SepalLengthCm'])

# box plot
sns.boxplot(x='Species',y='PetalLengthCm',data=df)

# histograma
sns.histplot(data=df,x='SepalLengthCm',stat='frequency')

# Combining Box and Strip Plots
ax=sns.boxplot(x='Species',y='SepalLengthCm',data=df)
ax=sns.stripplot(x='Species',y='SepalLengthCm',
                 data=df,jitter=True,edgecolor='gray')

# grafico violin
sns.violinplot(x='Species',y='SepalLengthCm',data=df,size=6)

# regresiones entre todos
sns.pairplot(data=df,kind='scatter',hue='Species')

# mapa de calor
plt.figure(figsize=(7,4))
sns.heatmap(df.corr(),annot=True,cmap='summer')

# distribuciones
df.hist(edgecolor='black', linewidth=1.2)
fig=plt.gcf()
fig.set_size_inches(12,6)
plt.show()

# swaram plot
sns.set(style="whitegrid")
fig=plt.gcf()
fig.set_size_inches(10,7)
fig = sns.swarmplot(x="Species", y="PetalLengthCm", data=df)