# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 22:29:34 2021

@author: gabri
"""


import seaborn as sns
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# carga de datos
df_iris = sns.load_dataset("iris")
df = df_iris
df = df.drop(['species'],axis=1)


# grafico informacion sin estandarizar
sns.pairplot(df_iris, diag_kind="kde")

# calcular pca
npc = PCA()
npc.fit(df)

# grafica
fig, ax = plt.subplots()
x = np.array(['blue','green','red'])
colores = np.repeat(x,50)
colores
ax.set_xlabel = "PC1"
ax.set_ylabel = "PC2"
ax.scatter(npc['PC1'], npc['PC2'],c = colores)
