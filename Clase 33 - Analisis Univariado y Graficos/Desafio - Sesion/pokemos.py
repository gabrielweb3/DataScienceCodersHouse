'''
prpuesta de ejercicio de clases
analisis univariado pokemones6
'''

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# leo archivo
df = pd.read_csv('pokemon.csv')

# descripcion de datos
estadisticas = df.describe()
print(estadisticas)

# agrupo por pokemon
agrupacion_por_pokemos = df.groupby(['name']).unique()
