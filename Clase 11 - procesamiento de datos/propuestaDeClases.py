# -*- coding: utf-8 -*-
"""
Hacer PCA con el set de datos IRIS
Graficar la componente 1 vs la 2

"""

# Tratamiento de datos
# ==============================================================================
import numpy as np
import pandas as pd
import statsmodels.api as sm
import pandas.util.testing as tm

# Gráficos
# ==============================================================================
import matplotlib.pyplot as plt
import matplotlib.font_manager
from matplotlib import style
style.use('ggplot') or plt.style.use('ggplot')
import seaborn as sns

# Preprocesado y modelado
# ==============================================================================
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale

# Configuración warnings
# ==============================================================================
import warnings
warnings.filterwarnings('ignore')

df_iris = sns.load_dataset("iris")


df = df_iris
df_iris = df_iris.drop(['species'],axis=1)

# scatter_matrix(df, alpha = 0.6, figsize = (10, 10), diagonal = 'kde')
# grafico informacion sin estandarizar
sns.pairplot(df_iris, diag_kind="kde")

# estandarizo y vuelvo a graficar
scaler = StandardScaler()
datos_standard = scaler.fit_transform(df_iris)

datos_standard = pd.DataFrame(datos_standard, columns= df_iris.columns)

sns.pairplot(datos_standard, diag_kind="kde")

# hacemos PCA
pca = PCA()
pca.fit(datos_standard)

# explicacion de varianza
pca.explained_variance_ratio_

proyecciones = pca.transform(X=datos_standard)

df_PCA = pd.DataFrame(proyecciones, columns = ['PC-1', 'PC-2', 'PC-3', 'PC-4'])

fig, ax = plt.subplots()

ax.scatter(df_PCA['PC-1'],df_PCA['PC-2'])
ax.set_xlabel('PC-1')
ax.set_ylabel('PC-2')

sns.pairplot(df_PCA, diag_kind="kde")

fig, ax = plt.subplots()
x = np.array(['blue','green','red'])
colores = np.repeat(x,50)
colores
ax.set_xlabel = "PC1"
ax.set_ylabel = "PC2"
ax.scatter(pca['PC1'], pca['PC2'],c = colores)