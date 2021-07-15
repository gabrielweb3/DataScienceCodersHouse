# -*- coding: utf-8 -*-
"""
estructuras de datos
@author: gabri
"""
# ejercicio 1
# Listas
print('Listas')
A = [1,2,3]
B = [4,5,6]
print('A+B: ',A + B)
print('A x 2',A*2)
# Tuplas
print('Tuplas')
A = (1,2,3)
B = (4,5,6)
print('A+B: ',A+B)
print('A x 2: ',A*2 )

# ejercicio 2
autos = {'Auto':['Gol','Clio','Fiesta','Renault'],'Marca':['Ford','Volbagen']}
print('Mal escrito: ',autos['Marca'][1])
autos['Marca'][1] = 'Volkswagen'
print('Corregido: ',autos['Marca'][1])

# ejercicio 3
lista = [0,1,2,3,4,5,6,7,8,9]
print(lista[1::2])
print(lista[::-1])