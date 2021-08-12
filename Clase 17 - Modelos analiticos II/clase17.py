"""
Clase 17
Modelos analiticos 2
11/8
"""
"""
# aprendizaje supervisado
# 1 - clasificacion

arbol de decision
desventajas, tiende al overfitting, por lo que el modelo al predecir nuevos casos estima con el mismo indice de acierto
se ven influencias por los outliers, creando arboles con ramas muy profundas que no predice bien para nuevos casos
crear arboles demasiado complejos puede conllevar a que no se adapten bien a los nuevos datos
se pueden crear arboles sesgados si una de las clases es mas numerosa que otro, si hay desbalance de clases

ejemplo
3 clases

seleccion multicategorica



"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

