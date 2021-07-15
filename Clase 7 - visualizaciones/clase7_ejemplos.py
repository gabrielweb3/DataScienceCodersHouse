# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 20:59:57 2021

@author: gabri
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# cambio estilo de graficos
mpl.style.use('bmh')


#graficos de lineas
fig,ax = plt.subplots()
# ax.plot([0,1,2,3,4,5,6],[1,5,2,4,8,9,2])
ax.plot([1,5,2,4,8,9,2])
# equivalentemente

#graficos de puntos
pesos = [42.8, 43.3, 42. , 44. , 44.1, 43.5, 48.1, 48.9, 47.7,46.9,50.4,
       52.7, 51.8, 54.5, 54.2, 56.9, 55.4, 55.5, 57.1, 58.3, 63.7, 58.8,
       64.6, 60.2, 64. , 63.8, 61.4, 66.3, 64.7, 63.9, 69.3, 67.9, 65.2,
       70.8, 70.5, 69.3, 75.3, 75.5, 78.2, 78. , 73.2, 78. , 80.1, 78.2,
       76. , 81.5, 79.4, 81.8, 81.8, 84.1]
alturas = [149. , 149. , 149.9, 156.8, 150.6, 155.4, 151. , 162. , 165.,
       157.8, 164.4, 160.1, 160.8, 163.8, 175.2, 162. , 159.5, 159.2,
       169.8, 166.7, 179.4, 180.6, 163.3, 178.8, 176.3, 184.8, 181. ,
       170.5, 184.1, 187.1, 187.1, 177.7, 184.5, 190.3, 196. , 192.1,
       200.4, 201.8, 187.5, 202.1, 200.3, 208.8, 204.6, 193.5, 200.9,
       196.8, 213.1, 204.8, 215.5, 210.2] 
fig,ax = plt.subplots()
ax.scatter(alturas, pesos, alpha=0.7)
ax.title('Altura vs Peso de 50 alumnos')
ax.set_xlabel('Altura (cm.)')
ax.set_ylabel('Peso (kg.)')