# -*- coding: utf-8 -*-
"""
3-En este link se provee un archivo con los resultados
    de la Encuesta de Sueldos de Openqube de Febrero 2020''
    (https://sueldos.openqube.io/encuesta-sueldos-2020.02/ ),
    Calcular y comparar media y mediana de los sueldos netos.
    https://docs.google.com/spreadsheets/u/0/d/15Stnum8mI9QdwF9ZiPFCe3_qG9XyZyNzmOSIDunzPjA/view
"""

import pandas as pd
import numpy as np

# cargo los datos
data = pd.read_excel('Salarios Openqube.xlsx')
# limpio los datos, ya que se observa que la columna que voy a usar no tiene un
# unico tipo de datos y no permite usar metodos numpy directamente
salarios = pd.to_numeric(data['Salario mensual BRUTO (en tu moneda local)'],errors='coerce')
# de la manera que esta usado el metodo to_numeric, convierte los tipos de datos
# no numericos en valores nan, por eso se utiliza el metodo dropna() para limpiar
# finalmente los datos que se utilizaran para el calculo
salarios = salarios.dropna()

# calculo mediana y media con metodos numpy
# no es necesario convertir la variable salarios a datos numpy ya que esta 
# libreria puede trabajar directamente con datos del tipo pandas
mediana_salarios = np.median(salarios)
media_salarios = np.mean(salarios)
print('Mediana de los salarios:',mediana_salarios)
print('Media de los salarios:',media_salarios)

# debido a que la media de los salarios me parecio un numero poco coherente
# calculo la mediana y la media directamente
# mediana
print('Mediana 2:',sorted(salarios)[int(len(salarios)/2)])
# media
print('Media 2:',sum(salarios)/len(salarios))
# los valores coinciden, asi que se prosigue con el analisis

# la media y la mediana de los sueldos son muy diferentes, y tiene bastante sentido
# que lo sean, ya que los sueldos no siguen una distribucion normal, donde la media y 
# la mediana coinciden