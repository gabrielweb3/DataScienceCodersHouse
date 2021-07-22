"""
Clase 11
data science
"""
# apuntes
# como detectar valores outliers, fuera de rangos
# con el boxplot los detectamos al toque
# dentro de la caja caen el 50% de los datos 

# analisis de componentes principales sirve para reduccion de dimensionalidades

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

USArrests = sm.datasets.get_rdataset("USArrests", "datasets")
datos = USArrests.data
print(datos.head(4))

from pandas.plotting import scatter_matrix
import seaborn as sns

df = datos
# scatter_matrix(df, alpha = 0.6, figsize = (10, 10), diagonal = 'kde')
sns.pairplot(datos, diag_kind="kde")

# estandarizacion de datos
scaler = StandardScaler()
datos_standard = scaler.fit_transform(datos)

datos_standard = pd.DataFrame(datos_standard, columns= datos.columns)
sns.pairplot(datos_standard, diag_kind="kde")

# calculamos PCA componentes principales
pca = PCA()
pca.fit(datos_standard)

print(PCA())

# explicacion de varianza
pca.explained_variance_ratio_
print(pca.explained_variance_ratio_)

# Opcional el uso de pipe
## Entrenamiento modelo PCA con escalado de los datos
## ==============================================================================
#pca_pipe = make_pipeline(StandardScaler(), PCA())
#pca_pipe.fit(datos)

# Se extrae el modelo entrenado del pipeline
#modelo_pca = pca_pipe.named_steps['pca']
#modelo_pca.explained_variance_ratio_
proyecciones = pca.transform(X=datos_standard)

df_PCA = pd.DataFrame(proyecciones, columns = ['PC-1', 'PC-2', 'PC-3', 'PC-4'])

fig, ax = plt.subplots()

ax.scatter(df_PCA['PC-1'],df_PCA['PC-2'])
ax.set_xlabel('PC-1')
ax.set_ylabel('PC-2')

sns.pairplot(df_PCA, diag_kind="kde")


