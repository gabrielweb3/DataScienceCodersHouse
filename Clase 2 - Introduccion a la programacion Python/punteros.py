# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 21:51:08 2021

@author: gabri
"""

x = 1
type(x)

print(x)

x = 'hola'

print(x)

x = [1,2,3]

y = x

print(y,x)

x.append(4)

print(y,x) # aca esta la cosa
"""
x no es una variable, es la referencia a una variable
cuando yo digo y = x, lo que digo es que la referencia al lugar de memoria
donde esta x, es igual a la referencia al valor de memoria donde esta y
"""
x = 5 

print(x,y)