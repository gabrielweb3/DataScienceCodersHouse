
"""
Ejercicio 1:
Importar Pandas y Numpy y crear el siguiente 
diccionario
dicc = {'First Score':[100, 90, np.nan, 95],
             'Second Score': [30, 45, 56, np.nan],
             'Third Score':[np.nan, 40, 80, 98]}
Ejercicio 2:
	Transformar en un dataframe y
    luego aplicar el método dropna(). Que paso?
Ejercicio 3: 
Crear un nuevo dataframe que rellena
 los na con 0. Se animan a rellenarlo 
 con la media de cada columna?

"""
import pandas as pd
import numpy as np


# ejercicio 1
dicc = {'First Score':[100, 90, np.nan, 95],
        'Second Score': [30, 45, 56, np.nan],
        'Third Score':[np.nan, 40, 80, 98]}

# ejercicio 2
df = pd.DataFrame(dicc) 
df = df.dropna()

# ejercicio 3
df2 = pd.DataFrame(dicc)
df2 = df2.fillna(df2.median())
# for columna in df2.columns:
#     df2[columna] = df2[columna].fillna(df2[columna].median())