"""
entrega propuesta
1-Generar un array aleatorio de 100 elementos.
    Calcular la mediana correspondiente.
    
2-Recordar los ejercicios con funciones para cálculo de factorial
    y suma de serie de la Clase 02. Repetir ambos ejercicios, pero ahora utilizar
    las nuevas operaciones aprendidas con los ndarrays.

3-En este link se provee un archivo con los resultados
    de la Encuesta de Sueldos de Openqube de Febrero 2020''
    (https://sueldos.openqube.io/encuesta-sueldos-2020.02/ ),
    Calcular y comparar media y mediana de los sueldos netos.
    https://docs.google.com/spreadsheets/u/0/d/15Stnum8mI9QdwF9ZiPFCe3_qG9XyZyNzmOSIDunzPjA/view
"""

import numpy as np
import urllib2 as url2

# genero array global para utilizarlo en otros ejecicios
array_1 = []
url = 'https://docs.google.com/spreadsheets/u/0/d/15Stnum8mI9QdwF9ZiPFCe3_qG9XyZyNzmOSIDunzPjA/view' 
contenido_url = url2.urlopen(url)

# 1 - array aleatorio de 100 elementos
def generar_array_random():
    print('Ejercicio 1...')
    global array_1
    array_1 = np.random.randint(100,size=101)
    return print('Mediana del array =',np.median(array_1))
generar_array_random()

# 2 - calculo de factorial y suma de todos los elementos
print('############################################################')
print('Ejercicio 2, parte A...')
def factorial(n):
    return print(f'{n}! = ',np.math.factorial(n))
factorial(0)
factorial(3)
print('############################################################')

print('Ejercicio 2, parte B...')
def suma_serie(serie):
    return print('Resultado de la suma de la serie:',serie.sum())
suma_serie(array_1)
suma_serie(np.arange(101))

# 3 - calculo y comparacion de media y mediana de los sueldos netos
print('############################################################')
def comparar_media_mediana():
    # importo pandas para levantar archivo excel 
    import pandas as pd
    global contenido_url
    print('Ejercicio 3...')
    print('cargando...')
    # cargo los datos
    data = pd.read_excel('Salarios Openqube.xlsx')
    # data = pd.read_excel(contenido_url)
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
    media_salarios = round(np.mean(salarios),2)
    # debido a que la media de los salarios me parecio un numero poco coherente
    # calculo la mediana y la media directamente
    # mediana
    Mediana_2 = sorted(salarios)[int(len(salarios)/2)]
    # media
    Media_2 = round(sum(salarios)/len(salarios),2)
    # comparo los calculos
    print(mediana_salarios == Mediana_2)
    print(media_salarios == Media_2)    
    # los valores coinciden, asi que se prosigue con el analisis
    
    # la media y la mediana de los sueldos son muy diferentes, y tiene bastante sentido
    # que lo sean, ya que los sueldos no siguen una distribucion normal, donde la media y 
    # la mediana coinciden
    return print(f" Mediana de los salarios: {mediana_salarios}\n",f"Media de los salarios: {media_salarios}")
comparar_media_mediana()

