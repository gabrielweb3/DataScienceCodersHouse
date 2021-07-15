# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 19:40:04 2021

@author: gabri
"""

import numpy as np

Npa = np.array(range(10))
print(Npa)

np_ceros = np.zeros(10,dtype=int)
print(np_ceros)

np_one = np.ones(10)
print(np_one)

np_relleno = np.full(10,265)
print(np_relleno)

np_range = np.arange(10)
print(np_range)

print(np.random.random(10)) # genera random entre 0 y 1

np_random_dimensions = np.random.randint(10,size=(3,4))
print(np_random_dimensions)

print(np_ceros.ndim)
print(np_random_dimensions.ndim)

print(np_ceros.shape)
print(np_random_dimensions.shape)

print(np_random_dimensions.size)

print(np_ceros.dtype)

print(np_random_dimensions.itemsize)

print(np_ceros.nbytes)
print(np_ceros.nbytes)

rango = range(1,11)
Np_diez_numeros = np.array(rango)
print(Np_diez_numeros)
print(Np_diez_numeros[0])
print(Np_diez_numeros[4])
print(Np_diez_numeros[-1])
print(Np_diez_numeros[-2])

print(np_random_dimensions)
print(np_random_dimensions[2,1])

print(Np_diez_numeros[:4])
print(Np_diez_numeros[3:])
print(Np_diez_numeros[4:7])
print(Np_diez_numeros[::2])
print(Np_diez_numeros[::-2])

print(np_random_dimensions[2,])
print(np_random_dimensions[:2,:2])
print(np_random_dimensions[2,3])


