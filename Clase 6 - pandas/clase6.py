# -*- coding: utf-8 -*-
"""
manipulacion de datos con pandas
"""
import pandas as pd

df_lluvia_arhivo = pd.read_csv('pune_1965_to_2002.csv')

indice = list(df_lluvia_arhivo.Year)
print(indice)

columnas = df_lluvia_arhivo.columns[1:]
print(columnas)

valores = df_lluvia_arhivo.values[:,1:]

df_lluvias = pd.DataFrame(valores,index=indice,columns=columnas)
print(df_lluvias)

# suma de precipitaciones por mes
print(df_lluvias.sum())
# promedio de precipitaicones por ano
print(df_lluvias.mean(axis='columns'))

# describir dataframe
print(df_lluvias.describe().round(1))
print(df_lluvias.T.describe().round(1))
