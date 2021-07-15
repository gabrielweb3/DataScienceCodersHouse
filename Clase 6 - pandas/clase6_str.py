# -*- coding: utf-8 -*-
"""
manipulacion de datos con pandas
"""
import pandas as pd

presidentes_archivos = pd.read_csv('us_presidents 2.csv')

presidentes_nombres = presidentes_archivos['president']
print(presidentes_nombres)

print(presidentes_nombres.str.upper())
print(presidentes_nombres.str.len())
print(presidentes_nombres.str.startswith(['J']))
print(presidentes_nombres.str.split())