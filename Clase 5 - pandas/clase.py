"""
clase 5, introduccion a pandas

"""
import pandas as pd
import numpy as np

numeros = range(50,70,2)

numeros_serie = pd.Series(numeros)
print(numeros_serie)
print(numeros_serie[2])

print(numeros_serie.index)
print(numeros_serie.values)

numeros_en_texto = ['primero','segundo','tercero',
                    'cuarto','quinto','sexto',
                    'septimo','octavo','noveno',
                    'decimo']
numero_serie_2 = pd.Series(numeros,index=numeros_en_texto)
print(numero_serie_2)


modelos = ['A4 3.0 Quattro 4dr manual',
 'A4 3.0 Quattro 4dr auto',
 'A6 3.0 4dr',
 'A6 3.0 Quattro 4dr',
 'A4 3.0 convertible 2dr']
peso = [3583, 3627, 3561, 3880, 3814]
precios = ['$33,430', '$34,480', '$36,640', '$39,640', '$42,490']

Autos_peso = pd.Series(peso,index=modelos)
Autos_precio = pd.Series(precios,index=modelos)
print(Autos_peso)
print(Autos_precio)

Autos = pd.DataFrame({'Peso':Autos_peso,'Precio':Autos_precio})
print(Autos)

Ajedrez_64 = np.arange(1,65).reshape(8,8)
Ajedrez_df = pd.DataFrame(Ajedrez_64,
                          columns=range(1,9),
                          index=['A','B','C','D','E','F','G','H'])
print(Ajedrez_df) 

print('SELECCIONANDO SERIE')
print(numero_serie_2['quinto'])
print(numero_serie_2.loc['quinto'])
print(numero_serie_2.iloc[5])

print(Autos.index)
print(Autos.columns)
print(Autos.values)
print(Autos['Peso'])
print(Autos.values[1])
print(Autos.loc['A4 3.0 Quattro 4dr auto'])

print(Autos.loc[Autos.Peso >= 3600],'Precio')

print('operaciones con pandas')
transpuesta = Autos.T
print(transpuesta)

print('Funciones vectorizadas')
numeros3 = range(51,70,2)
numeros_serie_3 = pd.Series(numeros3,index=numeros_en_texto)
print(numeros_serie_3)

print('ufuncs sobre dataframes')
largo = [179,179,192,192,180]
Autos_2 = pd.DataFrame({'Peso':peso,'Largo':largo},index=modelos)
print(Autos_2)
print(Autos_2/Autos_2.iloc[0]*100)

print(numero_serie_2+numeros_serie_3)
# hacer lo mismo pero de otra manera
print(numero_serie_2.add(numeros_serie_3))

print('Conservacion de indices')
numeros_serie_2_porcion = numero_serie_2[4:7]
numeros_serie_3_porcion = numeros_serie_3[5:8]
print('suma de porciones')
print(numeros_serie_2_porcion+numeros_serie_3_porcion)

print('Valores faltantes')
print(numeros_serie_2_porcion.add(numeros_serie_3_porcion,fill_value=0))

valor_nan = np.nan
print(type(valor_nan))
print(2*valor_nan)
print(np.nanprod([2,valor_nan]))

print('trabajando con datos faltantes')
numeros_nan = numeros_serie_2_porcion+numeros_serie_3_porcion
print(numeros_nan)
print('isnull():',numeros_nan.isnull())
print('fillna(0):',numeros_nan.fillna(0))
print('dropna():',numeros_nan.dropna())