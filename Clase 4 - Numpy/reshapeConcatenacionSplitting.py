# -*- coding: utf-8 -*-
"""
reshape
concatenacion
splitting
"""
import numpy as np

print('Reshape')
ajedrez_64 = np.arange(1,65).reshape(8,8)
print(ajedrez_64) 

print('#############')
print('Splitting')
arrays_contatenados = np.array([0,6,5,4,3,1,8,0,0,3])
print(arrays_contatenados)

array_partido = np.split(arrays_contatenados,[2])
print(array_partido)

array_partido_2 = np.split(arrays_contatenados,[2,8])
print(array_partido_2)
parte_1, parte_2, parte_3 = array_partido_2
print(parte_1,parte_2,parte_3)

print('Corte vertical')
ajedrez_partido = np.hsplit(ajedrez_64,[4])
print(ajedrez_partido)

print ('Corte horizontal')
ajedrez_partido_2 = np.vsplit(ajedrez_64,[4])
print(ajedrez_partido_2)
                               

print('#############')
print('Concatenacion')

array_1 = np.random.randint(10,size=5)
array_2 = np.random.randint(10,size=5)
array_concatenado = np.concatenate([array_1,array_2])
print(array_concatenado)

print('Concatenacion vertical')
array_extra = np.array([[10],[20]])
print(array_extra)

print('array apilados verticalmente')
array_apilados_vetical =  np.vstack([array_extra,array_extra])
print(array_apilados_vetical)

print('array apilados horizontalmente')
array_apilados_horizontal = np.hstack([array_extra,array_extra])
print(array_apilados_horizontal)