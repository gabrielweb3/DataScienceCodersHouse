"""
ENTREGA 3
Series de tiempo con pandas

1-Con este archivo, construir un objeto de series de tiempo con el índice
    igual al año de los juegos olímpicos
2-Separar nombres de apellidos en dos columnas distintas usando la coma como separador
3-Obtener medidas resumen del conjunto de datos.
    Cuál es el país que ganó más medallas?
4-Construir una tabla que muestre cuántas medallas obtuvieron
    los hombres en total en cada año que se realizó el evento.

"""
import pandas as pd

# levantar archivo
medallero = pd.read_csv('Summer-Olympic-medals-1976-to-2008.csv',encoding='latin-1')


# datetimes de inicio y fin
inicio = pd.to_datetime(str(int(medallero['Year'][0])))
fin = pd.to_datetime(str(int(medallero['Year'][-1:])))
print('Inicio:',inicio,', Fin:',fin)
# rango de tiempo
rango_tiempo = pd.date_range(start=inicio,end=fin,periods=((medallero['Year'].max()-medallero['Year'].min())/4)+1)

# creo y limpio dataframe
anos_olimpicos = pd.DataFrame(index=rango_tiempo.year) # dataframe con index igual al ano de los JJOO
# anos_olimpicos = pd.DataFrame(medallero['Year'].unique())
# anos_olimpicos = anos_olimpicos.dropna()
# anos_olimpicos = anos_olimpicos.astype(int)

# dataframe con fechas de index
# medallero_2 = pd.DataFrame(index=anos_olimpicos[0])

# divir columna de atletas
medallero["Nombre"], medallero["Apellido"] = medallero["Athlete"].str.split(",", 1).str

# resumenes de datos

# pais con mas medallas 
# medallero['Year'] = medallero['Year'].astype(int)
medallas_pais = pd.DataFrame(medallero.groupby(['Country'])['Medal'].count()) 
medallas_pais = medallas_pais.sort_values('Medal',ascending=False)
print(medallas_pais.head(10))

# total de medallas de hombre por ano
medallas_genero = pd.DataFrame(medallero.groupby(['Gender','Year'])['Medal'].count())
# print(medallas_genero.index[0],medallas_genero['Medal'][0],'Medallas')