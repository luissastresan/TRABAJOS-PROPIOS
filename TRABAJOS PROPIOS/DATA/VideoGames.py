#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 08:35:53 2020

@author: luissastresanemeterio
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("VideoGamesSales.csv")

# Primer paso: 

## Inspeccionamos un poco el dataset
data.dtypes # La mayoria son datos numericos, pero hay alguno categórico
data.head(10) # Aparecen nan el las primeras filas de algunas variables
data.tail(10) # Tambien al final aparecen nan
data.describe

## Trabajamos los Nan
data.isnull().values.any() # Hay nan
data.isnull().sum().sum() # Tenemos 46716 valores desconocidos
data.isnull().sum() # Parece que las variables mas afectadas son la de puntuación

## Sustituimos los nan por la media de la columna en los datos que son numericos
## (Critic_Score, Critic_Count y User_Count)
data["Critic_Score"].fillna(value = data["Critic_Score"].mean(), inplace = True)
data["Critic_Count"].fillna(value = data["Critic_Count"].mean(), inplace = True)
data["User_Count"].fillna(value = data["User_Count"].mean(), inplace = True)

## La variable User_Score tiene valores nan y "tbd", que significa "no identificado". 
## Remplazo los Nan por 0 y los "tbd" tambien. Y casteo la columna para convertirla 
## en tipo float.
data["User_Score"].fillna(0, inplace = True)
data.replace({'tbd': '0'}, inplace = True)
data["User_Score"] = data["User_Score"].astype(float)

## Sustituimos los nan por None en las variables categoricas (Developer, Rating).
data["Developer"].fillna(value = "None", inplace = True)
data["Rating"].fillna(value = "None", inplace = True)

## Elimino el resto de Nan
data = data.dropna()

## Compruebo que no quedan nan
data.isnull().values.any()


# Analisis exploratorio de los datos

# Observamos que los juegos mas usados son los de accion con diferencia. Los que menos
# los de Puzzle.
genre = data["Genre"].value_counts().sort_values(ascending = False)
plt.figure(figsize = (8,6))
sns.barplot(y = genre.index, x = genre.values, orient = 'h')
plt.xlabel = 'Cantidad de juegos'
plt.ylabel = 'Genre'
plt.show()

# Las compañias que mas juegos lanzan son Electronic Arts, Activision, Namco
publisher = data["Publisher"].value_counts().sort_values(ascending = False).head(15)
plt.figure(figsize = (8,6))
sns.barplot(y = publisher.index, x = publisher.values, orient = 'h')
plt.xlabel = 'Cantidad de juegos'
plt.ylabel = 'Publisher'
plt.show()

# Y la plataforma que mas juegos lanza
platGenre = pd.crosstab(data.Platform, data.Genre) # Creo una tabla identificando cada plataforma el numero de juegos  de cada genero.
platGenreTotal = platGenre.sum(axis=1).sort_values(ascending = False) # Sumo cada fila para dar el total de juegos de cada plataforma.
plt.figure(figsize=(8,6))
sns.barplot(y = platGenreTotal.index, x = platGenreTotal.values, orient='h')
plt.ylabel = "Platform"
plt.xlabel = "Cantidad de juegos"
plt.show()

# Intentamos mostrarlo graficamente
platGenre['Total'] = platGenre.sum(axis=1)
popPlatform = platGenre[platGenre['Total']>1000].sort_values(by='Total', ascending = False)
neededdata = popPlatform.loc[:,:'Strategy']
maxi = neededdata.values.max()
mini = neededdata.values.min()
popPlatformfinal = popPlatform.append(pd.DataFrame(popPlatform.sum(), columns=['total']).T, ignore_index=False)
sns.set(font_scale=0.7)
plt.figure(figsize=(10,5))
sns.heatmap(popPlatformfinal, vmin = mini, vmax = maxi, annot=True, fmt="d")
plt.xticks(rotation = 90)
plt.show()

# Ahora representamos las 9 compañias que lanzan mas juegos y las comparamos 
# con el tipo de juego, para ver que tipo de juego lanza mas cada compañia,
subdata = data[data.Publisher.isin(["Electronic Arts", "Activision", "Namco Bandai Games",
                                    "Ubisoft", "Komani Digital Entertainment", "THO",
                                    "Nintendo", "Sony Computer Entertainment", "Sega"])]

grafica = sns.factorplot(
        x = "Publisher", 
        hue = "Genre",
        data = subdata,
        kind = "count", 
        size = 7, 
        aspect = 2.0,
        legend = False,
        ).set_axis_labels('Publisher', 'Frecuencia')

grafica.ax.legend(loc= 'upper right', bbox_to_anchor = (0.5, 1.1), shadow = False, ncol=3,
                  labels = ['Action','Sports', 'Misc', 'Role-Playing', 
                            'Shooter', 'Adventure', 'Racing', 'Platform', 'Simulation', 
                            'Fighting', 'Strategy','Puzzle'])


plt.show()


# Voy a indagar un poco dividiendo el dataset por decadas
subdata1 = data[data.Year_of_Release.isin([1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987,
                1988, 1989])]

subdata2 = data[data.Year_of_Release.isin([1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997,
                1998, 1999])]

subdata3 = data[data.Year_of_Release.isin([2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007,
                2008, 2009])]

subdata4 = data[data.Year_of_Release.isin([2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017,
                2018, 2019])]


# Veamos si hay correlación entre las ventas en cada zona, las globales y el genero del juego.
import sklearn.preprocessing as preprocessing

# Creamos una función para codificar las la base de datos.
def number_encode_features(data):
    result = data.copy()
    encoders = {}
    for column in result.columns:
        if result.dtypes[column] == 'object':
            encoders[column] = preprocessing.LabelEncoder()
            result[column] = encoders[column].fit_transform(result[column].astype('str'))
    return result, encoders

# Bien, vamos a trabajar con las observaciones mas recientes.
encoded_data, _ = number_encode_features(subdata4)

# Representamos la matriz de correlacion:
plt.figure(figsize = (10,10))

sns.heatmap(encoded_data.corr(), square=True)
plt.show()
# Podemos ver que las variables de ventas estan muy correlacionadas entre ellas, es algo logico.

# Ahora voy a intentar predecir el tipo de genero que tiene el videojuego por sus ventas globales.

# REGRESIÓN LINEAL
# Definimos mis variables dependientes e independientes
x = encoded_data.iloc[:, 9].values
y = encoded_data.iloc[:, 3].values

# Dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)

# Escalado de variables
x_train = x_train.reshape(-1, 1)
x_test = x_test.reshape(-1, 1)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)


# Crear modelo de Regresión Lienal Simple con el conjunto de entrenamiento
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(x_train, y_train)

# Predecir el conjunto de test
y_pred = regression.predict(x_test)

# Visualizar los resultados de entrenamiento
plt.scatter(x_train, y_train, color = "red")
plt.plot(x_train, regression.predict(x_train), color = "blue")
plt.title("Ventas vs Gérero del Videojuego (Conjunto de Entrenamiento)")
plt.xlabel("Genero del Videojuego")
plt.ylabel("Ventas Globales (en $)")
plt.show()

# Visualizar los resultados de test
plt.scatter(X_test, y_test, color = "red")
plt.plot(X_train, regression.predict(X_train), color = "blue")
plt.title("Sueldo vs Años de Experiencia (Conjunto de Testing)")
plt.xlabel("Años de Experiencia")
plt.ylabel("Sueldo (en $)")
plt.show()










