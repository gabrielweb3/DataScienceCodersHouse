# -*- coding: utf-8 -*-
"""
clase 18
aprendizaje no 
https://docs.google.com/presentation/d/1ZjvnwK9VY6zcW8_88bYF567TPVmhM-YSba-_5UbiZ9w/edit#slide=id.p18
diferencias
no supervisado
infieren patrones de un conjunto de datos sin referencia a resultados conocidos o etiquetados
eso significa que no se cuenta con variable Y, solo X
no tenemoos variable objetivo
no hay variables que ayuden a predecir la var de salida
todas las variables tienen la misma importancia
no hay posibilidades de predecir a la variable de salida
se busca interdependencia entre variables
funciona con datos no etiquetados, su proposito es nutalmente la exploracion
si el aprendizaje supervisado funciona bajo reglas claramente definidas
los algoritmos de ANS estan acostumbrados a:
    explorar ;a estructura de la imformacion y detectar patrones
    extraer ideas valiosas
    aumentar la eficacia del proceso de toma de decisiones en base a los patrones detectados
    
algoritmos:
    clustering o agrupamiento
    reglas de asociacion
    algoritmos de reduccion de dimensionalidad
    
clustering:
    se intenta encontrar en una estructura o un patron en una coleccion de datos no clasificados
    intentan encontrar grupos en los datos que compartan atributos en comun
    tanbueb varuabkles numericas para medir la distacia
    para trabajar con variables categoricas haciendolas variables dummy, con el uso
    de la tecnica de transformacion de datos ONE HOT ENCODING OHE
    se asignan valores binarios a datos categoricos, por ejemplo nacionalidad
tipos de clustering:
    jerarquicos: se generan sucesiones ordenads(jerarquicas) de conglomerados
    puede ser agrupando clusters pequenos en uno o mas grande, dividiendo
    grandes clusters en otros mas pequenos
    la estructura arbol se llama Dendograma
    tipos:
        aglomerativos(bottom-up), inicialmente cada instancia es un cluster
        las partes aglomerativas parten de un conjunto chico o de elementos 
        individuales y van juntando los elementos hasta quedarse con un numero
        de cluster que se considere optimo
        divisivos(top-down): incialmente las instancias estan en un solo cluster
        y se van dividiendo. las estategias divisivas, parten del conjunto de elementos
        completos y se van separando en grupos diferentes entre si, hasta quedarse
        con un numero de cluster que se considere optimo
    no jerarquico:
        la cantidad de clusters se escoge de antemano, y los registros se asignan
        a los clusters segun se cernacia. el algoritmo mas conocido es el 
        K-means
diferentes tipos de distancias en clustering:
    euclidiana, manhattan, minkowski, chebychev, cosine similarity, hamming
    cada una de estas distancias puden cambiar sustancialmente el resultado
    
reglas de asociacion:
    tienen un antecedente y un consecuente, izquierda y derecha, ambos lados de la regla son
    un conjunto de elementos
    si el conjunto de elementos de X es antecedente y conjunto de elementos Y
    es el consecuente, entonces la regla se escribe como X->Y
    {fideos, salsa}->{queso rallado}
    ventajas: el concepto es muy sencillo, su implementacion no requiere
    gran complejidad y tiene buena performance
    desventajas: se generan muchas reglas con un pequeno numero de elementos
    las reglas pueden ser ciclicas (A, B) → C, (A, C) → B y (B, C) → A. 
    Implementación e Python Apriori para análisis de asociación

reduccion de dimensionalidad:
    se busca reducir cantidad de features de un dataset, pero reteniendo
    la mayor cantidad de informacion
    se puede eliminar variables del dataset o se pueden aplicar transformaciones
    matematicas
    ejemplo sw globo terraqueo
    el objetivo es perder la menor cantidad de informacion posible
    se quiere obtener una variable(vector) que es transformacion lineal
    de otras variables
    algoritmos de aplicacion:
        PCA analisis de componentes principales
        Auto-encoders con redes neuronales
        MDS multidimensional scaning
        UMAP
        
    PCA: el metodo gira los datos de forma que, desde un punto de vista 
    estadistico no exista correlacion entre las caracteristicas rotadas, 
    pero que conserven la mayor cantidad posible de la varianza de los datos 
    originales
    reduce dimensionalidad, de un conjunto de datos, proyectandose sobre un 
    subespacio de menor dimension
"""

