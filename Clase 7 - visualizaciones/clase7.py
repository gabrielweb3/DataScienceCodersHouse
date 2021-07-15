# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 20:37:14 2021

@author: gabri
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# cambio estilo de graficos
mpl.style.use('bmh')

#graficar dos puntos noma
# a = [1,3]
# b = [2,4]

# # interfaz orientada a objetos
# fig,ax = plt.subplots()
# ax.plot(a,b)

# # interfaz orientada a estados
# # plt.plot(a,b)

# # etiquetas de ejes
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# # titulos
# ax.set_title('un grafico orientado a objetos')
# # leyenda
# ax.legend('a--')

# importa datos de lluvias
df_lluvias = pd.read_csv('pune_1965_to_2002 (1).csv')
print(df_lluvias.head())
df_lluvias = df_lluvias.set_index('Year')

# ejes
x = df_lluvias.index
y = df_lluvias.sum(axis='columns')

#defino objetos fig y ax
fig,ax = plt.subplots(figsize=(12,4))
ax.plot(x,y,label='Precipitaciones acumuladas')
# etiquetamos ejes, anadimos titulos e insertamos la leyenda
ax.set_xlabel('Ano')
ax.set_ylabel('Precipitaciones acumuladas(mm.)')
ax.set_title('Precipitaciones acumuladas por ano')
ax.legend()
#recorto bordes vacios
ax.set_xlim(df_lluvias.index[0], df_lluvias.index[-1])
# guardo figura
fig.savefig('precipitaciones_acumuladas.pdf')

# graficos de puntos
fig, ax = plt.subplots()  
mapeo_colores = ax.scatter(df_lluvias['Aug'],
                           df_lluvias['Sep'],
                           c = df_lluvias.index)
fig.colorbar(mapeo_colores)
ax.set_title('Precipitaciones Agosto-Setiembre')
ax.set_ylabel('Precipitaciones en Agosto (mm.)')
ax.set_ylabel('Precipitaciones en Setiembre (mm.)')
fig.savefig('comparacion_ago_set.pdf')

# graficos de barras
precipitaciones_acumuladas = df_lluvias.sum()
print(precipitaciones_acumuladas)
fig,ax = plt.subplots(figsize=(8,4))
precipitaciones_acumuladas = df_lluvias.sum()
ax.bar(df_lluvias.columns, precipitaciones_acumuladas)
ax.set_xlabel('Mes')
ax.set_ylabel('Precipitacion total (mm.)')
ax.set_title('Precipitaciones acumuladas desde 1965 a 2002')
fig.savefig('precipitaciones_acumuladas_bar.pdf')

# histograma
# df_lluvias.flatten()
fig,ax = plt.subplots(figsize=(8,4))
ax.hist(df_lluvias.values.flatten(),bins=10)
ax.set_title('Histograma de precipitaciones')
ax.set_xlabel('Intervalos de precipitaciones (mm.)')
ax.set_ylabel('Frecuencia absoluta')
fig.savefig('histograma.pdf')

# enriquecimientos de graficos
fig,ax = plt.subplots(figsize=(12,3))
ax.plot(df_lluvias.index, df_lluvias['Jan'],label='Precipitaciones de Enero')
ax.plot(df_lluvias.index, df_lluvias['Feb'],label='Precipitaciones de febrero',color='red')
# agragacion de maximo
maximo_enero = df_lluvias['Jan'].max()
maximo_febrero = df_lluvias['Feb'].max()
# grafico lineas horizontales con valores definidos
ax.axhline(maximo_enero,color='red',
           linestyle='--',alpha=0.5,
           linewidth=3,label='Maxima enero')
ax.axhline(maximo_febrero,color='red',
           linestyle=':',alpha=0.5,
           linewidth=3,label='Maxima febrero')
# etiqueras 
ax.set_xlabel('Ano')
ax.set_ylabel('Precipitaciones(mm.)')
ax.set_title('Precipitaciones de enero y febrero')
ax.set_xlim(df_lluvias.index[0],df_lluvias.index[-1])
ax.legend()

fig.savefig('graf_enriquecido.pdf')