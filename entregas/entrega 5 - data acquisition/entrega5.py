
"""
Entrega 5
data acquisition
adquisicion de json desde web
https://jsonplaceholder.typicode.com/users

1. escribir codigo que lea json https://jsonplaceholder.typicode.com/users
2. pasarlo a dataframe de pandas

"""
# librerias de manejo de datos
import pandas as pd
import json
import requests

# especifico enlace de api
url = 'https://jsonplaceholder.typicode.com/users'

# hago el request del enlace
response = requests.get(url)
# aparto el texto de la request
text = response.text

# creo un json con los diccionarios del texto de la request
json = json.loads(text)

# creo vectores para separar diccionario de direcciones
street = []; suite = []; city = []; zipcode = []; lat = []; long = []
# creo vectores para separar diccionario de compania
company_name = []; catch_frase = []; businnes = []

# 

for i in range(0,len(json)):
    # address
    street.append(json[i]['address']['street'])
    suite.append(json[i]['address']['suite'])
    city.append(json[i]['address']['city'])
    zipcode.append(json[i]['address']['zipcode'])
    lat.append(json[i]['address']['geo']['lat'])
    long.append(json[i]['address']['geo']['lng'])
    # company
    company_name.append(json[i]['company']['name'])
    catch_frase.append(json[i]['company']['catchPhrase'])
    businnes.append(json[i]['company']['bs'])
    
# creo dataframe con direcciones
columns = ['Street','Suite','City','ZipCode','Latitud','Longitud','Company Name','Catch Phrase','Business'] 
address_company = pd.DataFrame(list(zip(street,suite,city,zipcode,lat,long,company_name,catch_frase,businnes)),
                       columns = columns)
# borro vectores ya utilizados
del street,suite,city,zipcode,lat,long,company_name,catch_frase,businnes

# cargo dataframe con datos de json
users = pd.DataFrame(json)

# elimino columnas con diccionarios
users = users.drop([users.columns[4],users.columns[7]],axis=1)

# agrego dataframe creado y obtengo dataframe final
users = pd.concat([users,address_company],axis=1)


# usuarios = pd.DataFrame.from_dict(json)
