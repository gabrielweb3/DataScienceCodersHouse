"""

Desafio de clases


Ejercicio 1
definir un array z=np.array([1,2,3]) que ocurre?
que me puede decir de cada elemento

Ejercicio 2:
	Dado A = np.array([4,2,8]) y B = np.array([4,2,8]). Hacer A+B, A*B y A-B. Que ocurre.
 Que pasa si cambio A por A = np.array([4,2,8,3])


Ejercicio 3: 
	Definir un array c = np.array([1.,4,”a”,False]). Que ocurre?

"""

import numpy as np

# ejercicio 1 
A = np.array([1,2,3])
print(A)
for a in A:
    print(a)
    print(a.dtype)
    
    
# ejercicio 2 
A = np.array([4,2,8])
B = np.array([4,2,8])
print('A:',A,',','B:',B)
print(A+B)
print(A*B)
print(A-B)
A = np.array([4,2,8,3]) # objeto de dimension diferente
print(A+B)
print(A*B)
print(A-B)

# ejercicio 3
C = np.array([1.,4,'a',False])
print(C)