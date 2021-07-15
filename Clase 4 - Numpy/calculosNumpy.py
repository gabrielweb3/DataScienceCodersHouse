# -*- coding: utf-8 -*-
"""
Calculos con elementos Numpy
"""
import numpy as np

array_aleatorio = np.random.randint(10,size=10)
print(array_aleatorio)

suma = 0
for i in array_aleatorio:
    suma+=i
promedio = suma/np.size(array_aleatorio)
print(promedio)

# diferentes operaciones
print('Suma:',array_aleatorio.sum()) # suma
print('Avg:',array_aleatorio.mean())# promedio
print('Max:',array_aleatorio.max()) # max
print('Mediana:',np.median(array_aleatorio)) # mediana
print('Std:',np.std(array_aleatorio)) # desviacion std
print('Varianza:',np.var(array_aleatorio)) # varianza

print('################')

arr1 = np.array([7,0,6,7,5])
arr2 = np.array([3,7,9,9,0])
print(arr1+5)
print(arr1+arr2)
print(np.add(arr1,arr2))

print('################')

print(np.matmul(arr1,arr2))

