# -*- coding: utf-8 -*-
"""
Desafio: PRACTICA CON FUNCIONES

1.Escribir una función para calcular el factorial de un número cualquiera.
2.Escribir una función para calcular la suma de una serie comenzando por un 
  número cualquiera y terminando en otro número que debe ser mayor al primero.

Alumno: Gabriel Olivera

Curso: Data Science
Academia: CoderHouse
Camada: 14125
Profesor: David Gabriel Algorta
Tutor: Gianluca Peretti
Año: 2021

"""
# 1
def calcular_factorial(numero):
    
    factorial = 1
    for n in range(1, numero+1):
        factorial *= n
    return print(f"{n}! = " , factorial)
    
calcular_factorial(5)
calcular_factorial(10)

# 2
def sumar_serie(n1,n2):
    resultado = 0
    
    if n1 < n2:
        for i in range(n1,n2+1):
            resultado+=i
        return print(f'Sumar todos los numeros desde {n1} hasta {n2} da como resultado:',resultado)
    else:
        return print('No se puede sumar la siguiente serie de numeros, escriba el numero menor en el primer lugar')

sumar_serie(1, 100)
sumar_serie(4,2)
