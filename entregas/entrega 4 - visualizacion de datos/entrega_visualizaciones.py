# -*- coding: utf-8 -*-
"""
Entrega SeaBorn

1. cargar archivo en python, realizar estadisticas descriptivas basicas
2. realizar histograma con los salarios. Que rangos de salarios son los mas 
    populares?
3. realizar grafico violin con los salarios discriminados por genero
4. graficar la serie de tiempo correspondiente a la fecha de contratacion
    (DateofHire)
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1 - cargar datos y realizar estadisticas basicas
# carga de datos
HR_dataset = pd.read_csv('HRDataset_v14.csv')
# calculo de estadisticas descriptivas basicas
print(HR_dataset.describe())

# 2 - histograma de salarios y rangos de salarios
Salarios = HR_dataset['Salary'].values.flatten()/1000

# Rangos de salarios
Rangos_salarios = HR_dataset['Salary'].value_counts(bins=20)
print(HR_dataset['Salary'].value_counts(bins=20))

# configuracion y ploteo de histograma
fig,ax = plt.subplots(figsize=(8,4))
# ax.hist(Salarios,bins=int(len(Salarios)/2),density=True)
ax.hist(Salarios,bins=20,density=True,
        lw=1, ec='grey',fc='orange')
ax.set_title('Histograma de Salarios')
ax.set_xlabel('Salarios (K$)')
ax.set_ylabel('Frecuencia absoluta')
ax.text(105, .025,f'Rango de salarios mas popular($): {Rangos_salarios.index[0]}\nCantidad de personas dentro del rango: {Rangos_salarios.max()}')
ax.grid(True)
ax.set_xlim(Salarios.min(),Salarios.max())
fig.savefig('histogramaSalarios.pdf')

# 3 - grafico de violin con salarios descriminados por genero
# categorical plots
ax = sns.catplot(data=HR_dataset, kind='violin', 
                 x='Sex', y='Salary', hue='Sex', 
                 split=True)
ax.set(xlabel='Genero', 
       ylabel='Salarios', 
       title='Distribucion de Salarios Segun Genero')
ax.savefig('salariossegungeneros.pdf')


# 4 - serie de tiempo correspondiente a la fecha de contratacion
HR_dataset['DateofHire'] = pd.to_datetime(HR_dataset['DateofHire'],yearfirst=True)
fig,ax = plt.subplots(figsize=(8,4))
# Se realiza una grafica con las series de EmpID y DateofHire ordenados,
# ya que el dato EmpID es un numero correlativo que crece dependiendo el DateofHire
# por lo tanto graficar estos valores ordenados, muestra una evolucion de la cantidad 
# de empleados en el tiempo
ax.plot(HR_dataset['DateofHire'].sort_values(ascending=True),HR_dataset['EmpID'].sort_values(ascending=True))
ax.set_title('Cantidad de Empleados en el tiempo')
ax.set_xlabel('A??o')
ax.set_ylabel('ID de Empleados')
ax.grid(True)
fig.savefig('lineatemporalDateofHire.pdf')
