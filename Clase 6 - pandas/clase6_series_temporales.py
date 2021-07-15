# -*- coding: utf-8 -*-
"""
series de tiempo
"""
import pandas as pd

presidentes_archivos = pd.read_csv('us_presidents 2.csv')

presidentes_nombres = presidentes_archivos['president']

fecha = pd.to_datetime('03/01/2020',dayfirst=True)
print(fecha)

fin = pd.to_datetime('10/01/2020',dayfirst=True)
fechas_1 = pd.date_range(start=fecha,end=fin)
print(fechas_1)

fechas_2 = pd.date_range(start=fecha,periods=8)
print(fechas_2)

fechas_3 = pd.date_range(start=fecha,periods=8,freq='M')
print(fechas_3)

mes_inicio = fecha.strftime('%Y-%m')
print(mes_inicio)

fechas_4 = pd.date_range(start=mes_inicio,periods=8,freq='M')
print(fechas_4)

cuanto_tiempo = fechas_3[7] - fechas_3[0]
print(cuanto_tiempo)

print('fecha a periodo')
cuanto_tiempo_meses = fechas_3[7].to_period('M')-fechas_3[0].to_period('M')
print(cuanto_tiempo_meses)

fechas_presidentes_orig = presidentes_archivos['start']
print(fechas_presidentes_orig)
print(type(fechas_presidentes_orig))

fechas_presidentes = pd.DatetimeIndex(fechas_presidentes_orig)
print(fechas_presidentes)

Serie_presidentes = pd.Series(presidentes_nombres.values,index=fechas_presidentes)
print(Serie_presidentes)

