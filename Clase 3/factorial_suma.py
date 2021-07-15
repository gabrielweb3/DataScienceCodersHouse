# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 22:06:27 2021

@author: gabri
"""

# factorial
def factorial(n):
    resultado = 1 
    for i in range(1,n+1):
        resultado = resultado*i
    return print(f"Resultado de {n}! = " , resultado)
factorial(5)

# suma hasta el 100
def suma_N_hasta_100():
    suma = 0
    for i in range(1,101):
        suma += i
    print('Suma de numeros N del 1 al 100: ',suma)
    
suma_N_hasta_100()