# procesamiento de datos
import pandas as pd
import numpy as np
# visualizacion de datos
import seaborn as sns
import matplotlib.pyplot as plt
# clasificaicon de datos
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score

# definicion de dataframe principal
ag01 = pd.DataFrame()

# carga de datos
def carga_de_datos():
    global ag01
    ag01 = pd.read_csv('ARI_datos_crudos.csv')

# informacion basica de datos
def informacion_y_descripcion():
    print(ag01.info())
    print(ag01.describe())

#################### limpieza de datos ##############################
def limpieza_basica():
    global ag01
    # elimino columna de indice de los datos crudos
    ag01 = ag01.drop([ag01.columns[0]],axis=1)
    ag01 = ag01.drop([ag01.columns[6]],axis=1) # columna de disponibilidad no sirve para nada
    
    # renombro columas para facilitar analisis
    # ag01.columns = ag01.rename(columns={ag01.columns[0]: 'Fecha',
    #                             ag01.columns[1]: 'Velocidad',
    #                             ag01.columns[2]: 'Velocidad_std',
    #                             ag01.columns[3]: 'Temperatura',
    #                             ag01.columns[4]: 'Potencia',
    #                             ag01.columns[5]: 'Estado de sistema',
    #                             ag01.columns[6]: 'Disponibilidad',}, inplace=True)
    
    # columns = ['Fecha','Velocidad','Velocidad_std',
    #                         'Temperatura','Potencia','Estado de sistema',
    #                         'Disponibilidad']
    # for col in range(0,len(ag01.columns)):
    #     ag01.columns[col] = columns[col]
    
    # convierto columnas que necesito del dataframe a valores numericos
    for col in range(1,len(ag01.columns)):
        ag01[ag01.columns[col]] = pd.to_numeric(ag01[ag01.columns[col]],errors='coerce')
    
    # convierto columna 'fecha' en tipo datetime
    ag01['fecha'] = pd.to_datetime(ag01['fecha'])
    
    # se filtran nan para poder hacer los analisis de las variables mas ordenadamente
    ag01 = ag01.dropna()

##############################################################################
######################## ANALISIS UNIVARIADO #################################
##############################################################################
# Se hara analisis unicamente de las variables mas importantes, que son las que se
# utilizaran para el modelado
def analisis_univariado():
    def univariado_velocidad():
        global ag01
        # velocidad de viento
        media_de_velocidad_todo_el_perdiodo = ag01[ag01.columns[1]].mean()
        mediana_de_velocidad_todo_el_perdiodo = ag01[ag01.columns[1]].median()
        moda_de_velocidad_todo_el_perdiodo = ag01[ag01.columns[1]].mode()
        print('Media, mediana y moda de velocidad de viento para todo el conjunto de datos')
        print('Media de velocidad de viento: ',media_de_velocidad_todo_el_perdiodo)
        print('Mediana de velocidad de viento: ',mediana_de_velocidad_todo_el_perdiodo)
        print('Moda de velocidad de viento: ',moda_de_velocidad_todo_el_perdiodo)
        
        # sns.set_theme(style="whitegrid")
        
        line_velocidad = sns.lineplot(x=ag01['fecha'],y=ag01[ag01.columns[1]],palette='pastel')
        line_velocidad.set_title('Evolucion temporal de velocidad de viento')
        line_velocidad.set_ylabel('Velocidad de viento')
        line_velocidad.set_xlabel('Fecha')
        line_velocidad.grid(True)
        # line_velocidad.set_xlim(0,25)
        
        dist_velocidad = sns.distplot(ag01[ag01.columns[1]])
        dist_velocidad.set_title('Distribucion de Weibull de la velocidad de viento')
        dist_velocidad.set_ylabel('Frecuencia relativa')
        dist_velocidad.set_xlabel('Velocidad de viento')
        dist_velocidad.grid(True)
        
        box_velocidad = sns.boxplot(y=ag01[ag01.columns[1]])
        box_velocidad.set_title('Distribucion de velocidad de viento')
        box_velocidad.set_ylabel('Velocidad de viento')
    
    def univariado_potencia():
        global ag01
        
        # potencia electrica
        media_de_potencia_todo_el_perdiodo = ag01[ag01.columns[4]].mean()
        mediana_de_potencia_todo_el_perdiodo = ag01[ag01.columns[4]].median()
        moda_de_potencia_todo_el_perdiodo = ag01[ag01.columns[4]].mode()
        print('Media, mediana y moda de Potencia electrica para todo el conjunto de datos')
        print('Media de Potencia electrica: ',media_de_potencia_todo_el_perdiodo)
        print('Mediana de Potencia electrica: ',mediana_de_potencia_todo_el_perdiodo)
        print('Moda de Potencia electrica: ',moda_de_potencia_todo_el_perdiodo)
        
        line_potencia = sns.lineplot(x=ag01['fecha'],y=ag01[ag01.columns[4]],palette='pastel')
        line_potencia.set_title('Evolucion temporal de la potencia electrica')
        line_potencia.set_ylabel('Potencia Electrica')
        line_potencia.set_xlabel('Fecha')
        line_potencia.grid(True)
        
        dist_potencia = sns.distplot(ag01[ag01.columns[4]])
        dist_potencia.set_title('Distribucion la Potencia Electrica')
        dist_potencia.set_ylabel('Frecuencia relativa')
        dist_potencia.set_xlabel('Potencia Electrica')
        dist_potencia.grid(True)
        
        box_potencia = sns.boxplot(y=ag01[ag01.columns[4]])
        box_potencia.set_title('Distribucion de la Potencia Electrica')
        box_potencia.set_ylabel('Potencia Electrica')
    
    def univariado_temperatura():
        global ag01
        # temperatura del ambiente 
        media_de_temperatura_todo_el_perdiodo = ag01[ag01.columns[3]].mean()
        mediana_de_temperatura_todo_el_perdiodo = ag01[ag01.columns[3]].median()
        moda_de_temperatura_todo_el_perdiodo = ag01[ag01.columns[3]].mode()
        print('Media, mediana y moda de temperatura ambiente para todo el conjunto de datos')
        print('Media de temperatura ambiente: ',media_de_temperatura_todo_el_perdiodo)
        print('Mediana de temperatura ambiente: ',mediana_de_temperatura_todo_el_perdiodo)
        print('Moda de temperatura ambiente: ',moda_de_temperatura_todo_el_perdiodo)
        
        line_temperatura = sns.lineplot(x=ag01['fecha'],y=ag01[ag01.columns[3]],palette='pastel')
        line_temperatura.set_title('Evolucion temporal de la temperatura ambiente')
        line_temperatura.set_ylabel('Temperatura Ambiente')
        line_temperatura.set_xlabel('Fecha')
        line_temperatura.grid(True)
        
        dist_temperatura = sns.distplot(ag01[ag01.columns[3]])
        dist_temperatura.set_title('Distribucion la Temperatura Ambiente')
        dist_temperatura.set_ylabel('Frecuencia relativa')
        dist_temperatura.set_xlabel('Temperatura Ambiente')
        dist_temperatura.grid(True)
        
        box_temperatura = sns.boxplot(y=ag01[ag01.columns[3]])
        box_temperatura.set_title('Distribucion de la Temperatura Ambiente')
        box_temperatura.set_ylabel('Temperatura Ambiente')
        
    univariado_velocidad()
    univariado_potencia()
    univariado_temperatura()

##############################################################################
######################## ANALISIS BIVARIADO ##################################
##############################################################################
# Se hara analisis unicamente de la potencia y la velocidad, que son las que se
# mas importantes para calcular la curva de potencia
# visualizaciones de curva de potencia
def analisis_multivariado():
    def multivariado_potencia_velocidad():
        global ag01
        sns.set_theme(color_codes=True)
        curva_de_potencia = sns.scatterplot(x=ag01[ag01.columns[1]],y=ag01[ag01.columns[4]],
                                            marker="+",
                                            color='b')
        curva_de_potencia.set_title('Curva de Potencia (todo el conjunto de datos son filtros)')
        curva_de_potencia.set_ylabel('Potencia')
        curva_de_potencia.set_xlabel('Velocidad')                            
                
        regresion_lineal = sns.regplot(x=ag01[ag01.columns[1]],y=ag01[ag01.columns[4]],
                                       color='r',marker="+")
        print('Coeficiente de correlacion en curva de potencia sin filtrar')
        print('R2 = 0.686510')
    
    multivariado_potencia_velocidad()
    
    analisis_de_componentes_principales()

##############################################################################
###################### DATA CLEANING #########################################
##############################################################################
def filtrado_de_datos():
    global ag01, ag01_filtrado
    
    ag01_filtrado = ag01.copy(deep=True)
    
    # filtros por norma y caracteristicas de maquina
    
    # filtros de velocidad
    ag01_filtrado = ag01[ag01[ag01.columns[1]]>=3.5]
    ag01_filtrado = ag01_filtrado[ag01_filtrado[ag01_filtrado.columns[1]]<23]
    
    # filtros de potencia
    ag01_filtrado = ag01_filtrado[ag01_filtrado[ag01_filtrado.columns[4]]>350]
    
    # filtros de estado de sistema
    ag01_filtrado = ag01_filtrado[ag01_filtrado[ag01_filtrado.columns[5]]!=0]
    
    # vectores de rango 
    velocidades_minimas = np.arange(3.5,25,0.4)
    for i in range(0,len(velocidades_minimas)):
        velocidades_minimas[i] = (velocidades_minimas[i]**2)/5
    
    # filtros de banda envolvente
    # # defino bandas de potencia y de velocidad
    zona_1 = [0,50]; velocidad_min_max_1 = [1.5,6]
    zona_2 = [51,100]; velocidad_min_max_2 = [1.6,6]
    zona_3 = [101,150]; velocidad_min_max_3 = [1.8,6]
    zona_4 = [151,200]; velocidad_min_max_4 = [2,7]
    zona_5 = [201,250]; velocidad_min_max_5 = [2.3,7]
    zona_6 = [251,300]; velocidad_min_max_6 = [2.5,6.6]
    zona_7 = [301,350]; velocidad_min_max_7 = [3.5,6.6]
    zona_8 = [351,400]; velocidad_min_max_8 = [3.7,6.8]
    zona_9 = [401,450]; velocidad_min_max_9 = [4.3,7]
    zona_10 = [451,500]; velocidad_min_max_10 = [4.67,7.3]
    zona_11 = [501,550]; velocidad_min_max_11 = [5.1,7.45]
    zona_12 = [551,600]; velocidad_min_max_12 = [5.32,7.5]
    zona_13 = [601,650]; velocidad_min_max_13 = [5.59,7.72]
    zona_14 = [651,700]; velocidad_min_max_14 = [5.74,7.82]
    zona_15 = [701,750]; velocidad_min_max_15 = [6.07,8.1]
    zona_16 = [751,800]; velocidad_min_max_16 = [6.28,8.3]
    zona_17 = [801,850]; velocidad_min_max_17 = [6.426,8.5]
    zona_18 = [851,900]; velocidad_min_max_18 = [6.66,8.66]
    zona_19 = [901,950]; velocidad_min_max_19 = [6.8,8.77]
    zona_20 = [951,1000]; velocidad_min_max_20 = [6.9,8.92]
    zona_21 = [1001,1050]; velocidad_min_max_21 = [7.09,9]
    zona_22 = [1051,1100]; velocidad_min_max_22 = [7.17,9.2]
    zona_23 = [1101,1150]; velocidad_min_max_23 = [7.25,9.35]
    zona_24 = [1151,1200]; velocidad_min_max_24 = [7.35,9.55]
    zona_25 = [1201,1250]; velocidad_min_max_25 = [7.46,9.72]
    zona_26 = [1251,1300]; velocidad_min_max_26 = [7.6,9.76]
    zona_27 = [1301,1350]; velocidad_min_max_27 = [7.7,9.8]
    zona_28 = [1351,1400]; velocidad_min_max_28 = [7.76,9.9]
    zona_29 = [1401,1450]; velocidad_min_max_29 = [7.95,10.05]
    zona_30 = [1451,1500]; velocidad_min_max_30 = [8.02,10.2]
    zona_31 = [1501,1550]; velocidad_min_max_31 = [8.15,10.3]
    zona_32 = [1551,1600]; velocidad_min_max_32 = [8.24,10.4]
    zona_33 = [1601,1650]; velocidad_min_max_33 = [8.34,10.53]
    zona_34 = [1651,1700]; velocidad_min_max_34 = [8.42,10.6]
    zona_35 = [1701,1750]; velocidad_min_max_35 = [8.48,10.74]
    zona_36 = [1751,1800]; velocidad_min_max_36 = [8.62,10.82]
    zona_37 = [1801,1850]; velocidad_min_max_37 = [8.75,10.9]
    zona_38 = [1851,1900]; velocidad_min_max_38 = [8.77,11.2]
    zona_39 = [1901,1950]; velocidad_min_max_39 = [8.79,11.7]
    zona_40 = [1951,2050]; velocidad_min_max_40 = [9.038,25]
    
    # # filtro condicional para bandas 
    ag01_filtrado = ag01_filtrado[(ag01_filtrado[ag01_filtrado.columns[4]] >= zona_2[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_2[1]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_2[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_2[1])
                                  | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_3[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_3[1]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_3[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_3[1])
                                  | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_4[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_4[1]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_4[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_4[1])
                                  | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_5[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_5[1]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_5[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_5[1])
                                  | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_6[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_6[1]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_6[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_6[1])
                                  | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_7[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_7[1]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_7[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_7[1])
                                  | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_8[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_8[1]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_8[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_8[1])
                                  | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_9[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_9[1]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_9[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_9[1])
                                  | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_10[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_10[1]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_10[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_10[1])
                                
                                  | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_11[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_11[1]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_11[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_11[1])
                                  | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_12[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_12[1]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_12[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_12[1])
                                  | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_13[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_13[1]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_13[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_13[1])
                                  | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_14[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_14[1]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_14[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_14[1])
                                  | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_15[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_15[1]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_15[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_15[1])
                                  | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_16[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_16[1]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_16[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_16[1])
                                  | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_17[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_17[1]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_17[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_17[1])
                                  | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_18[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_18[1]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_18[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_18[1])
                                  | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_19[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_19[1]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_19[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_19[1])
                                  
                                  | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_20[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_20[1]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_20[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_20[1])
                                  | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_21[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_21[1]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_21[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_21[1])
                                  | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_22[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_22[1]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_22[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_22[1])
                                  | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_23[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_23[1]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_23[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_23[1])
                                  | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_24[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_24[1]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_24[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_24[1])
                                  | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_25[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_25[1]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_25[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_25[1])
                                  | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_26[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_26[1]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_26[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_26[1])
                                  | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_27[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_27[1]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_27[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_27[1])
                                  | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_28[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_28[1]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_28[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_28[1])
                                  
                                  | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_29[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_29[1]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_29[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_29[1])
                                  | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_30[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_30[1]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_30[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_30[1])
                                  | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_31[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_31[1]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_31[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_31[1])
                                  | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_32[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_32[1]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_32[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_32[1])
                                  | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_33[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_33[1]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_33[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_33[1])
                                  | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_34[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_34[1]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_34[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_34[1])
                                  | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_35[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_35[1]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_35[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_35[1])
                                  | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_36[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_36[1]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_36[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_36[1])
                                  | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_37[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_37[1]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_37[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_37[1])
                                  | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_38[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_38[1]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_38[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_38[1])
                                  | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_39[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_39[1]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_39[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_39[1])
                                  | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_40[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_40[1]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_40[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_40[1])
                                    ]
    
    # reseteo index
    ag01_filtrado = ag01_filtrado.reset_index(drop=True)
    
    # creo columna en dataframe principal que marca con un true los datos que pertenecen a la curva de potencia filtrada
    ag01['dentro de curva'] = ag01_filtrado[ag01_filtrado.columns[4]] >= zona_2[0] & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_2[1])& (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_2[0])& (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_2[1])| (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_3[0])& (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_3[1])& (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_3[0])& (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_3[1]) | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_4[0]) & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_4[1]) & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_4[0])& (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_4[1])| (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_5[0]) & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_5[1]) & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_5[0]) & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_5[1])| (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_6[0]) & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_6[1]) & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_6[0]) & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_6[1]) | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_7[0]) & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_7[1]) & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_7[0]) & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_7[1])| (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_8[0]) & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_8[1]) & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_8[0]) & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_8[1])| (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_9[0]) & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_9[1]) & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_9[0]) & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_9[1])| (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_10[0]) & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_10[1]) & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_10[0]) & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_10[1]) | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_11[0]) & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_11[1]) & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_11[0]) & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_11[1])| (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_12[0]) & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_12[1]) & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_12[0]) & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_12[1])| (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_13[0]) & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_13[1]) & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_13[0]) & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_13[1])| (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_14[0]) & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_14[1]) & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_14[0]) & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_14[1])| (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_15[0]) & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_15[1]) & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_15[0]) & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_15[1])| (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_16[0]) & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_16[1]) & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_16[0]) & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_16[1])| (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_17[0])  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_17[1]) & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_17[0]) & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_17[1]) | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_18[0]) & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_18[1]) & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_18[0]) & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_18[1])| (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_19[0]) & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_19[1]) & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_19[0]) & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_19[1]) | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_20[0]) & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_20[1]) & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_20[0]) & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_20[1])| (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_21[0]) & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_21[1]) & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_21[0]) & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_21[1]) | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_22[0]) & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_22[1]) & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_22[0]) & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_22[1])| (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_23[0]) & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_23[1]) & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_23[0]) & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_23[1])| (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_24[0]) & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_24[1]) & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_24[0]) & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_24[1])| (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_25[0]) & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_25[1]) & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_25[0]) & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_25[1])| (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_26[0])  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_26[1]) & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_26[0]) & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_26[1])| (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_27[0]) & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_27[1]) & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_27[0]) & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_27[1]) | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_28[0]) & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_28[1]) & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_28[0]) & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_28[1]) | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_29[0]) & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_29[1]) & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_29[0]) & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_29[1]) | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_30[0]) & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_30[1]) & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_30[0]) & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_30[1])| (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_31[0]) & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_31[1]) & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_31[0]) & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_31[1]) | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_32[0])  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_32[1])  & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_32[0]) & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_32[1]) | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_33[0]) & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_33[1]) & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_33[0]) & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_33[1])| (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_34[0]) & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_34[1]) & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_34[0]) & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_34[1]) | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_35[0]) & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_35[1]) & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_35[0]) & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_35[1]) | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_36[0]) & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_36[1]) & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_36[0]) & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_36[1]) | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_37[0]) & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_37[1]) & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_37[0]) & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_37[1])| (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_38[0]) & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_38[1]) & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_38[0]) & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_38[1])| (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_39[0])  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_39[1])  & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_39[0]) & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_39[1])| (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_40[0]) & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_40[1]) & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_40[0]) & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_40[1])
    
    # clasificacion de datos
    # cambio datos true por 1
    ag01['dentro de curva'] = ag01['dentro de curva'].map({ag01['dentro de curva'].unique()[0]:1,ag01['dentro de curva'].unique()[1]:0})
    # y como los datos que no pertenecen a la curva estan marcados con un nan, los cambio por 0

    # # filtro de datos consecutivos para valores numericos
    # datos_consecutivos = 3
    
    # for var in range(1,4):    
 
    #     # filtro de valores consecutivos
    #     filtrados = ag01.groupby((ag01[ag01.columns[var]].shift() != ag01[ag01.columns[var]]).\
    #                           cumsum()).filter(lambda x: len(x) >= datos_consecutivos)  
    #     # esta linea filtra finalmente los datos convirtiendolos de nueva a df
    #     ag01[ag01.columns[var]] = np.where(ag01.index.isin(filtrados.index), np.nan, ag01[ag01.columns[var]][:])
        
      
# visualizacion despues de filtrado
def visualizacion_de_filtrados():
    global ag01,ag01_filtrado
    curva_de_potencia_filtrada = sns.scatterplot(x=ag01_filtrado[ag01_filtrado.columns[1]],
                                                 y=ag01_filtrado[ag01_filtrado.columns[4]],
                                                 marker="+",
                                                 color='g')
    curva_de_potencia_filtrada.set_title('Curva de Potencia (todo el conjunto de datos con filtros)')
    curva_de_potencia_filtrada.set_ylabel('Potencia')
    curva_de_potencia_filtrada.set_xlabel('Velocidad')               
    curva_de_potencia_filtrada.grid(True)             
    print('Coeficiente de correlacion en curva de potencia sin filtrar')
    print('R2 = 0.686510')
    
    regresion_lineal = sns.regplot(x=ag01_filtrado[ag01_filtrado.columns[1]],
                                    y=ag01_filtrado[ag01_filtrado.columns[4]],
                                    color='g',marker="+")
    print('Coeficiente de correlacion de curva de potencia despues de filtrado')
    print('R2 = 0.923241')
    
    
def clasificacion_con_regresion_logistica():
    global ag01
    
    # features = [ag01.columns[1],ag01.columns[2],ag01.columns[5]]
    # features = [ag01.columns[1],ag01.columns[2],ag01.columns[3],ag01.columns[4],ag01.columns[5]]
    features = [ag01.columns[1],ag01.columns[2],ag01.columns[4],ag01.columns[5]]
    # features = [ag01.columns[1],ag01.columns[2]]
    # features = [ag01.columns[1],ag01.columns[2],ag01.columns[5]]
    target = ['dentro de curva']
    
    #Seleccionamos todas las columnas
    X = ag01[features]
    
    #Defino los datos correspondientes a las etiquetas
    y = ag01[target]
    
    ########## IMPLEMENTACIÓN DE REGRESIÓN LOGÍSTICA ##########
    
    from sklearn.model_selection import train_test_split
    
    #Separo los datos de "train" en entrenamiento y prueba para probar los algoritmos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    #Se escalan todos los datos
    from sklearn.preprocessing import StandardScaler
    escalar = StandardScaler()
    X_train = escalar.fit_transform(X_train)
    X_test = escalar.transform(X_test)
    
    #Defino el algoritmo a utilizar
    from sklearn.linear_model import LogisticRegression
    
    algoritmo = LogisticRegression(random_state=1, solver='liblinear',class_weight='balanced')
    
    #Entreno el modelo
    algoritmo.fit(X_train, y_train)
    
    #Realizo una predicción
    y_pred = algoritmo.predict(X_test)
    
    #Verifico la matriz de Confusión
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import plot_confusion_matrix
    # muestro graficamente matriz de confusion 
    confusion_matrix(y_test, y_pred)
    
    matriz = confusion_matrix(y_test, y_pred)
    print('Matriz de Confusión:')
    print(matriz)
    
    matriz_LR = plot_confusion_matrix(algoritmo, X_test, y_test,
                                      cmap='cividis')  
    plt.show()
    
    
    # coeficiente de modelo
    beta_0 = algoritmo.intercept_ # El beta 0
    beta_1 = algoritmo.coef_[0][0] # El coeficiente beta_1
    beta_2 = algoritmo.coef_[0][1] # El coeficiente beta_2
    print('Beta 0: ',beta_0)
    print('Beta 1: ',beta_1)
    print('Beta 2: ',beta_2)
    
    #Calculo la precisión del modelo
    from sklearn.metrics import precision_score
    
    precision = precision_score(y_test, y_pred)
    print('Precisión del modelo:')
    print(precision)
    
    #Calculo la exactitud del modelo
    from sklearn.metrics import accuracy_score
    
    exactitud = accuracy_score(y_test, y_pred)
    print('Exactitud del modelo:')
    print(exactitud)
    
    #Calculo la sensibilidad del modelo
    from sklearn.metrics import recall_score
    
    sensibilidad = recall_score(y_test, y_pred)
    print('Sensibilidad del modelo:')
    print(sensibilidad)
    
    #Calculo el Puntaje F1 del modelo
    from sklearn.metrics import f1_score
    
    puntajef1 = f1_score(y_test, y_pred)
    print('Puntaje F1 del modelo:')
    print(puntajef1)
    
    #Calculo la curva ROC - AUC del modelo
    from sklearn.metrics import roc_auc_score
    
    roc_auc = roc_auc_score(y_test, y_pred)
    print('Curva ROC - AUC del modelo:')
    print(roc_auc)
    
    print('Precisión del modelo:', precision)
    print('Exactitud del modelo:', exactitud)
    print('Sensibilidad del modelo:', sensibilidad)
    print('Puntaje F1 del modelo:', puntajef1)
    print('Curva ROC - AUC del modelo:', roc_auc)
    
    
def clasificacion_con_knn():
    from sklearn.metrics import accuracy_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import plot_confusion_matrix
    from sklearn import preprocessing
    global ag01
    
    random_seed = 2
    # Lista de features que vamos a considerar 
    features = [ag01.columns[1],ag01.columns[2],ag01.columns[3],ag01.columns[4],ag01.columns[5]]
    # features = [ag01.columns[1],ag01.columns[2],ag01.columns[4],ag01.columns[5]]
    # features = [ag01.columns[1],ag01.columns[2]]
    # features = [ag01.columns[1],ag01.columns[2],ag01.columns[5]]
    # variable a predecir
    target = ['dentro de curva']
    
    # Construcción de la matriz de features
    X = ag01[features].to_numpy()
    # print('Matriz de entradas ',X)
    # Construcción del vector a predecir
    y = ag01[target].to_numpy()
    # print('Vector a predecir: ',y)
    
    # Creacion de las matrices de entrenamiento y testeo. 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state=random_seed)
    print('Dimensión de la matriz de features para entrenamiento: {}'.format(X_train.shape))
    print('Dimensión de la matriz de features para testeo: {}'.format(X_test.shape))
    
    # scaler_train = preprocessing.MinMaxScaler()
    # X_train_scaled = scaler_train.fit_transform(X_train)
    # X_test_scaled = scaler_train.transform(X_test)
    
    # Normalizamos en train
    scaler_train = preprocessing.StandardScaler().fit(X_train)
    X_train_scaled = scaler_train.transform(X_train)
    # normalizo el test
    X_test_scaled = scaler_train.transform(X_test)
    
    # probar otros números para k
    # probar otras distancias, ej: euclidean, minkowski, manhattan 
    # probar dar mas peso a los vecinos de un orden superior: weights = 'distance'
    knn = KNeighborsClassifier(n_neighbors=2, metric='manhattan',algorithm='auto') 
    knn.fit(X_train_scaled, y_train)
    
    y_pred_train = knn.predict(X_train_scaled)
    accuracy_train =  accuracy_score(y_pred_train, y_train)
    print('El accuracy en el conjunto de train es', accuracy_train)
    
    # realizo prediccion con test set
    prediccion_test = knn.predict(X_test_scaled)
    accuracy_test =  accuracy_score(prediccion_test, y_test)
    print('El accuracy en el conjunto de test es', accuracy_train)
    # realizo comparacion de prediccion y datos de test
    y_test_arreglado = []
    for i in range(0,len(y_test)):
        y_test_arreglado.append(y_test[i][0])
    
    # creo vectores para contar y comparar prediccion y test
    cantidad_positivos_test = []
    cantidad_negativos_test = []
    cantidad_positivos_pred = []
    cantidad_negativos_pred = []
    
    # test_prediction = knn.predict(X_test_scaled)
    
    y_pred_T = []
    for i in range(0,len(y_pred_train)):
        y_pred_T.append([y_pred_train[i]])
        
    # plot_confusion_matrix(knn, prediccion_test, y_test_arreglado)  
    plot_confusion_matrix(knn, y_pred_T, y_train)  
    plt.show()

# analisis de componenetes principales
def analisis_de_componentes_principales():
    global ag01
    
    # defino x y
    # features = [ag01.columns[1],ag01.columns[2],ag01.columns[3],ag01.columns[4],ag01.columns[5]]
    features = [ag01.columns[1],ag01.columns[2]]
    # features = [ag01.columns[1],ag01.columns[2],ag01.columns[3],ag01.columns[4],ag01.columns[5]]
    target = ['dentro de curva']
    
    #Seleccionamos todas las columnas
    x = ag01[features]
    
    #Defino los datos correspondientes a las etiquetas
    y = ag01[target]
    # x = ag01.iloc[:,1:5].values
    # y = ag01.iloc[:,6].values
    
    # divido datos en conjuntos de datos de train y test
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)
    
    # estandarizacion y escalado
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    y_test = sc.transform(x_test)
    
    # aplico PCAxtest
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    x_train = pca.fit_transform(x_train)
    x_test = pca.transform(x_test)
    
    # Explicación de la varianza
    # Creamos un vector con el porcentaje de influencia de la varianza 
    # para las dos variables resultantes del conjunto de datos
    explained_variance = pca.explained_variance_ratio_
    print('Matriz de varianza: ',explained_variance)
    
    # realizo modelo de clasificacion luego de realizado el PCA
    from sklearn.linear_model import LogisticRegression
    clasificador = LogisticRegression(random_state=0)
    clasificador.fit(x_train, y_train)
    
    y_pred = clasificador.predict(x_test)
    
    #Calculo la precisión del modelo
    from sklearn.metrics import precision_score
    
    precision = precision_score(y_test, y_pred)
    print('Precisión del modelo:')
    print(precision)
    
    # matriz de confusion
    # from sklearn.metrics import confusion_matrix
    from sklearn.metrics import plot_confusion_matrix
    
    matriz = confusion_matrix(x_test, y_pred)   
    
    plot_confusion_matrix(clasificador, x_test, y_pred)
                           
    plt.show()

# clasificacion con arbol de decision
def clasificacion_arbol_de_decision():
    global ag01
    
    # features = [ag01.columns[1],ag01.columns[2],ag01.columns[3],ag01.columns[4],ag01.columns[5]]
    features = [ag01.columns[1],ag01.columns[2],ag01.columns[4],ag01.columns[5]]
    # features = [ag01.columns[1],ag01.columns[2]]
    # features = [ag01.columns[1],ag01.columns[2],ag01.columns[5]]
    
   # Lista de features que vamos a considerar 
    # features = [ag01.columns[1],ag01.columns[2],ag01.columns[3],ag01.columns[4],ag01.columns[5]]
    # variable a predecir
    target = ['dentro de curva']
    
    #Seleccionamos todas las columnas
    X = ag01[features]
    
    #Defino los datos correspondientes a las etiquetas
    y = ag01[target]
    
    ########## IMPLEMENTACIÓN DE ÁRBOLES DE DECISIÓN CLASIFICACIÓN ##########
    
    from sklearn.model_selection import train_test_split
    
    #Separo los datos de "train" en entrenamiento y prueba para probar los algoritmos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    #Se escalan todos los datos
    from sklearn.preprocessing import StandardScaler
    escalar = StandardScaler()
    X_train = escalar.fit_transform(X_train)
    X_test = escalar.transform(X_test)
    
    #Defino el algoritmo a utilizar
    #Arboles de decisión
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.tree import DecisionTreeRegressor
    # algoritmo = DecisionTreeClassifier(criterion = 'entropy', random_state=1) # 65.8520
    # algoritmo = DecisionTreeClassifier(criterion = 'entropy', random_state=0) # 66,419
    # algoritmo = DecisionTreeClassifier(criterion = 'entropy', random_state=2) # 65,9747
    # algoritmo = DecisionTreeClassifier(criterion = 'gini') # 66,276
    # algoritmo = DecisionTreeClassifier(criterion = 'entropy')# 66,47
    # algoritmo = DecisionTreeClassifier(max_depth = 3,random_state = 123) # 62,6104
    algoritmo = DecisionTreeClassifier(max_depth= 5,criterion= 'entropy',random_state = 7)
    
    #Entreno el modelo
    algoritmo.fit(X_train, y_train)
    
    #Realizo una predicción
    y_pred = algoritmo.predict(X_test)
    
    # cross validation
    from sklearn.model_selection import cross_val_score
    cross_val_score(algoritmo,X_train,y_train,cv=3,scoring='recall')
    
    #Verifico la matriz de Confusión
    from sklearn.metrics import confusion_matrix
    matriz = confusion_matrix(y_test, y_pred)
    print('Matriz de Confusión:')
    print(matriz)
    
    #Calculo la precisión del modelo
    from sklearn.metrics import precision_score
    from sklearn.metrics import plot_confusion_matrix
    # muestro graficamente matriz de confusion 
    
    matriz = confusion_matrix(y_test, y_pred)
    print('Matriz de Confusión:')
    print(matriz)
    
    matriz_DT = plot_confusion_matrix(algoritmo, X_test, y_test,
                                      cmap='cividis')  
    plt.show()
    
    precision = precision_score(y_test, y_pred)
    print('Precisión del modelo:')
    print(precision)


# clasificacion por suport vector machine
def clasificacion_SVM():
    global ag01
    
    # Lista de features que vamos a considerar 
    features = [ag01.columns[1],ag01.columns[2],ag01.columns[3],ag01.columns[4],ag01.columns[5]]
    # variable a predecir
    target = ['dentro de curva']
    
    #Seleccionamos todas las columnas
    X = ag01[features]
    
    #Defino los datos correspondientes a las etiquetas
    y = ag01[target]
    
    from sklearn.model_selection import train_test_split
    
    #Separo los datos de "train" en entrenamiento y prueba para probar los algoritmos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # defino algoritmo svm
    from sklearn.svm import SVC
    algoritmo = SVC(kernel='linear')
    # entro modelo
    algoritmo.fit(X_train, y_train)
    
    # realizo prediccion
    y_pred = algoritmo.predict(X_test)
    
    # verifico matriz de confusion
    from sklearn.metrics import confusion_matrix
    matriz = confusion_matrix(y_test, y_pred)
    print('Matriz de confusion: ')
    print(matriz)
    # imprimo matriz
    from sklearn.metrics import plot_confusion_matrix
    matriz_LR = plot_confusion_matrix(algoritmo, y_test, y_pred,
                                      cmap='cividis')  
    plt.show()
    
    # precision del modelo
    from sklearn.metrics import precision_score
    precision = precision_score(y_test, y_pred)
    print('Precision del modelo: ')
    print(precision)

# algoritmo de clasificaicon gradiente estocastico
def clasificador_descenso_de_gradiente_estocastico():
    global ag01
    
   # Lista de features que vamos a considerar 
    features = [ag01.columns[1],ag01.columns[2],ag01.columns[3],ag01.columns[4],ag01.columns[5]]
    # variable a predecir
    target = ['dentro de curva']
    
    #Seleccionamos todas las columnas
    X = ag01[features]
    
    #Defino los datos correspondientes a las etiquetas
    y = ag01[target]
    
    from sklearn.model_selection import train_test_split
    
    #Separo los datos de "train" en entrenamiento y prueba para probar los algoritmos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    y_train_5 = (y_train == 5)
    y_test_5 = (y_test == 5)
    
    #Se escalan todos los datos
    from sklearn.preprocessing import StandardScaler
    escalar = StandardScaler()
    X_train = escalar.fit_transform(X_train)
    X_test = escalar.transform(X_test)    
    
    # implementacion de algoritmo
    from sklearn.linear_model import SGDClassifier

    sgd_clf = SGDClassifier(random_state =  42)
    sgd_clf.fit(X_train,  y_train_5)
    
    y_pred=sgd_clf.predict(y_test)
    
    #Verifico la matriz de Confusión
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import plot_confusion_matrix
    # muestro graficamente matriz de confusion 
    confusion_matrix(y_test, y_pred)
    
    matriz = confusion_matrix(y_test, y_pred)
    print('Matriz de Confusión:')
    print(matriz)
    
    matriz_LR = plot_confusion_matrix(sgd_clf, X_test, y_test,
                                      cmap='cividis')  
    plt.show()
    
    #cross validatios
    from sklearn.model_selection import cross_val_score
    cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
    
    #Calculo la precisión del modelo
    from sklearn.metrics import precision_score
    
    precision = precision_score(y_test, y_pred)
    print('Precisión del modelo:')
    print(precision)
    
    #Calculo la exactitud del modelo
    from sklearn.metrics import accuracy_score
    
    exactitud = accuracy_score(y_test, y_pred)
    print('Exactitud del modelo:')
    print(exactitud)
            
# algoritmo de clasificacion RANDOM FOREST
def clasificacion_por_random_forest():
    global ag01
    
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    
    features = [ag01.columns[1],ag01.columns[2],ag01.columns[3],ag01.columns[4],ag01.columns[5]]
    target = ['dentro de curva']
    
    #Seleccionamos todas las columnas
    X = ag01[features]
    
    #Defino los datos correspondientes a las etiquetas
    y = ag01[target]
    
    #Separo los datos de "train" en entrenamiento y prueba para probar los algoritmos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    #Se escalan todos los datos
    from sklearn.preprocessing import StandardScaler
    escalar = StandardScaler()
    X_train = escalar.fit_transform(X_train)
    X_test = escalar.transform(X_test)
    
    # #Creamos un arbol de decisión sencillo y lo fiteamos
    tree = DecisionTreeClassifier(random_state=11, max_depth=5)
    tree.fit(X_train, y_train)
    
    y_test_pred = tree.predict(X_test) #Prediccion en Test
    
    from sklearn.metrics import accuracy_score

    #Calculo el accuracy en Test
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print('% de aciertos sobre el set de evaluación 1:',test_accuracy)
    
    #Creamos un random forest!
    model = RandomForestClassifier(random_state=11, n_estimators=200,
                                   class_weight="balanced", max_features="log2")
    model.fit(X_train, y_train)
    
    y_test_pred = model.predict(X_test) #Prediccion en Test
    
    #Verifico la matriz de Confusión
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import plot_confusion_matrix
    # muestro graficamente matriz de confusion 
    confusion_matrix(y_test, y_test_pred)
    
    matriz = confusion_matrix(y_test, y_test_pred)
    print('Matriz de Confusión:')
    print(matriz)
    
    matriz_LR = plot_confusion_matrix(model, X_test, y_test_pred,
                                      cmap='cividis')  
    plt.show()
    
    
    #Calculo la exactitud del modelo
    from sklearn.metrics import accuracy_score
    
    #Calculo el accuracy en Test
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print('% de aciertos sobre el set de evaluación del random forest:',test_accuracy)


def prediccion_de_curva():
    global ag01, ag01_filtrado


    features = [ag01_filtrado.columns[1],ag01_filtrado.columns[2],ag01_filtrado.columns[3]]
    target = [ag01_filtrado.columns[4]]
    
    #Seleccionamos todas las columnas
    X = ag01_filtrado[features]
    
    #Defino los datos correspondientes a las etiquetas
    y = ag01_filtrado[target]
    
    # librerias para visualizacion
    import plotly.express as px
    from plotly.offline import plot
    
    datos_entrada = px.scatter(x=X[X.columns[0]],y=y[y.columns[0]])
    plot(datos_entrada,filename='datos entrada')
    
    
    #Separo los datos de "train" en entrenamiento y prueba para probar los algoritmos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # separo una variable de velocidad para graficar en el final
    velocidad_para_graficar = X_test[X_test.columns[0]].copy()
    
    #Se escalan todos los datos
    from sklearn.preprocessing import StandardScaler
    escalar = StandardScaler()
    X_train = escalar.fit_transform(X_train)
    X_test = escalar.fit_transform(X_test)
    
    import tensorflow as tf
    
    # definicion de modelo
    modelo = tf.keras.Sequential()
    
    # capas
    # 1 
    modelo.add(tf.keras.layers.Dense(units=100, input_shape=[3], activation='softplus'))
    # 2
    # modelo.add(tf.keras.layers.Dense(100, activation='sigmoid'))
    # 3
    modelo.add(tf.keras.layers.Dense(units=10, activation='relu'))
    # salida
    modelo.add(tf.keras.layers.Dense(units=1))
    
    # creo modelo de rn
    # modelo = tf.keras.Sequential([oculta1, oculta2, salida])
    
    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    #     initial_learning_rate=1e-2,
    #     decay_steps=10000,
    #     decay_rate=0.9)
    # optimizador = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
    
    # parametros de modelo
    optimizador = tf.keras.optimizers.Adam(0.01)
    funcion_de_perdidas = 'mean_squared_error'
    
    # compilacion de modelo
    modelo.compile(
        optimizer=optimizador,
        loss=funcion_de_perdidas,
        metrics=[tf.keras.metrics.Accuracy()])
    
    
    # entrenamiento y evaluacion del modelo
    print("Comenzando entrenamiento...")
    historial = modelo.fit(X_train, y_train, epochs=50, verbose=False)
    print("Modelo entrenado!")
    
    # visualizacion de evolucion de la funcion de perdida en funcion del epoch
    import matplotlib.pyplot as plt
    plt.xlabel("# Epoca")
    plt.ylabel("Magnitud de pérdida")
    plt.plot(historial.history["loss"])
    
    
    # prediccion y metricas del modelo
    prediccion = modelo.predict(X_test)
    
    # accuracy
    precision_modelo = tf.keras.metrics.Accuracy()
    precision_modelo.update_state(y_test, prediccion) 
    print('Accuracy del modelo:',precision_modelo.result())
    
    # preparacion datos para regresion
    real = []; predecido = [] 
    for i in range(0,len(y_test)):
        # real.append(y_test.values[i][0])
        predecido.append(prediccion[i][0])
    
    
    
    prediccion_potencia = px.scatter(x=velocidad_para_graficar,
                                     y=predecido)
    prediccion_potencia.update_layout(title_text='prediccion vs realidad')
    plot(prediccion_potencia,filename='prediccion vs realidad')


# funcion principal
def funcion_principal():
    carga_de_datos()
    informacion_y_descripcion()
    limpieza_basica()
    analisis_univariado()
    # analisis_multivariado()
    filtrado_de_datos()
    visualizacion_de_filtrados()
    # clasificacion_con_regresion_logistica()
    # clasificacion_con_knn()
    # analisis_de_componentes_principales()
    # clasificacion_arbol_de_decision()
    # clasificacion_por_random_forest()
    # clasificador_descenso_de_gradiente_estocastico() # todavia no se como funciona
    # clasificacion_SVM()  # demora horrores 

funcion_principal()