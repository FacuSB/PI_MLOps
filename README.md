<h1 align="center">Proyecto Individual MLOps N°1</h1>
<h2 align="center">Facundo Blanco</h2>
<p align="center">
<img src="https://user-images.githubusercontent.com/67664604/217914153-1eb00e25-ac08-4dfa-aaf8-53c09038f082.png"  height=300>
</p>

## Descripcion

En este proyecto, he desarrollado una API que proporciona consultas sobre datos de la plataforma Steam, implementando procesos de Data Engineering y Machine Learning Operations (MLOps). La API aborda la creación de un sistema de recomendación de videojuegos para usuarios, basándose en un conjunto de datos inicialmente poco maduro.
>## **Tabla de Contenidos**
>- [Introducción](#introducción)
>- [Desarrollo](#desarrollo)
>   - [ETL (Extracción, Transformación y Carga)](#1-etl-extracción-transformación-y-carga)
>   - [EDA (Análisis Exploratorio de Datos)](#2-eda-análisis-exploratorio-de-datos)
>   - [Datos Finales](#3-datos-finales)
>   - [Funciones](#4-funciones)
>   - [Despliegue](#5-despliegue)
>- [Uso](#uso)
>- [Documentación](#documentación)
>- [Fuente de Datos](#fuente-de-datos)
>- [Contacto](#contacto)

## **Introducción**
Este proyecto surge de la necesidad en Steam de crear un sistema de recomendación de videojuegos para usuarios. Inicialmente, los datos presentaban limitaciones en su madurez y estructura, lo que demandó un trabajo integral de Data Engineering.

La API resultante facilita consultas sobre información de videojuegos, desde el análisis de sentimientos en reseñas hasta funciones específicas como análisis de desarrolladores y recomendaciones personalizadas. Este README proporciona detalles sobre el desarrollo del proyecto, funciones clave y el despliegue de la API.

## **Desarrollo**

El proceso de desarrollo se divide en cinco partes esenciales:


- ### **1. ETL (Extracción, Transformación y Carga)**

    En esta fase, se procedió a realizar un procesamiento de datos mediante un notebook. Este notebook se encarga de leer los datos originales, transformarlos en dataframes y aplicar dos funciones fundamentales de limpieza desarrolladas específicamente para este proyecto. La primera función normaliza los datos de tipo fecha y numéricos, mientras que la segunda se encarga de limpiar datos vacíos y nulos, permitiendo un margen de tolerancia ajustable.

    De manera particular, para el dataset de reviews, se llevó a cabo una segmentación de la columna "reviews" en distintas columnas, lo que facilitará análisis posteriores. Además, se implementó una etapa de análisis de sentimientos utilizando la biblioteca de Python llamada TextBlob. Esta herramienta realiza una evaluación del texto recibido, asignando una calificación al sentimiento transmitido. Los sentimientos fueron clasificados en tres categorías: 2 para positivo, 1 para neutral y 0 para negativo.

    Finalmente, los dataframes resultantes de este proceso de tratamiento fueron convertidos en archivos CSV, preparándolos para ser utilizados en la siguiente etapa del proyecto.

    [Notebook ETL](Data\ETL.ipynb).

- ### **2. EDA (Análisis Exploratorio de Datos)**

    Durante la fase de Análisis Exploratorio de Datos (EDA), se llevó a cabo un estudio de los tipos de datos presentes en cada columna, así como de las relaciones entre ellas y la comunicación entre los distintos datasets. Se realizaron análisis individuales a cada uno de los datasets, donde se procedió a modificar ciertos datos con el objetivo de mejorar la comprensión de los mismos.

    Adicionalmente, se abordaron y corrigieron errores presentes en los datasets, se eliminaron duplicados y nulos que habían pasado desapercibidos durante la etapa de Extracción, Transformación y Carga (ETL).

    Para obtener una visión más clara de la interrelación de los datos entre los distintos datasets, se generaron gráficos. Estos gráficos se utilizaron para identificar y confirmar cómo se comunican los datos entre las diferentes fuentes, proporcionando información valiosa para etapas posteriores del proyecto.

    La notebook correspondiente a esta fase no solo ofrece un análisis detallado, sino que también devuelve datasets curados. Estos conjuntos de datos optimizados están preparados para ser utilizados en las próximas fases del proyecto, principalmente para luego hacer la construccion de la api.

    [Notebook EDA](Data\EDA.ipynb).

- ### **3. Datos Finales**

    El último notebook de esta fase tiene como objetivo la creación de un dataset final. Este dataset, tiene como finalidad facilitar y agilizar el desarrollo de las funciones en las etapas posteriores del proyecto.

    La confección de este dataset final se basa en la combinación y consolidación de la información relevante proveniente de los análisis EDA y las transformaciones aplicadas en fases anteriores. Se ha buscado garantizar la coherencia, calidad y simplicidad de los datos, con el propósito de optimizar la eficiencia en el desarrollo de las funciones que se implementarán en la API.

    [Notebook Datos_Finales](Data\Datos_Finales.ipynb).


- ### **4. Funciones**

    Para cumplir con los requisitos de la API, se desarrollaron un total de siete funciones, cada una con propósitos específicos:

    ##### 1. `developer(developer: str) -> Dict[str, Union[str, Dict[int, int], Dict[int, str]]]`

    Esta función se encarga de proporcionar información específica sobre un desarrollador de videojuegos. Al ingresar el nombre del desarrollador como parámetro, la función devuelve un diccionario que incluye el nombre del desarrollador, la cantidad de juegos lanzados por año y el porcentaje de contenido gratuito por año.

    ##### 2. `userdata(User_id: str) -> Dict[str, Union[str, str, str, str, str]]`

    La función `userdata` ofrece detalles sobre el comportamiento de un usuario en la plataforma. Al ingresar el ID de usuario como parámetro, la función devuelve un diccionario que incluye el nombre del usuario, el dinero gastado, el porcentaje de recomendación según las reviews, y la cantidad de items que posee.

    ##### 3. `UserForGenre(genero: str) -> Dict[str, Union[str, Dict[str, str], List[Dict[str, Union[str, int]]]]]`

    La función `UserForGenre` identifica el usuario que ha acumulado más horas jugadas para un género específico. Al ingresar el género como parámetro, la función devuelve un diccionario que incluye el nombre del usuario con más horas jugadas y una lista detallada de las horas jugadas por año.

    ##### 4. `best_developer_year(año: int) -> List[Dict[str, Union[str, str]]]`

    La función `best_developer_year` devuelve el top 3 de desarrolladores con juegos más recomendados por usuarios para un año específico. Al ingresar el año como parámetro, la función devuelve una lista con información detallada sobre los desarrolladores más destacados.

    ##### 5. `developer_reviews_analysis(desarrolladora: str) -> Dict[str, List[Dict[str, Union[str, int]]]]`

    La función `developer_reviews_analysis` analiza las reseñas de usuarios para una desarrolladora específica. Al ingresar el nombre de la desarrolladora como parámetro, la función devuelve un diccionario con el nombre de la desarrolladora y una lista de la cantidad total de registros de reseñas categorizados con análisis de sentimiento positivo o negativo.

    ##### 6. `recomendacion_juego(id_producto: str) -> List[Dict[str, str]]`

    La función `recomendacion_juego` proporciona una lista de 5 juegos recomendados similares al producto ingresado. Al ingresar el ID del producto como parámetro, la función devuelve una lista con información detallada sobre los juegos recomendados.

    ##### 7. `recomendacion_usuario(id_usuario: str) -> List[Dict[str, str]]`

    La función `recomendacion_usuario` ofrece una lista de 5 juegos recomendados para un usuario específico. Al ingresar el ID del usuario como parámetro, la función devuelve una lista con información detallada sobre los juegos recomendados.
    
    *Nota: En el código de cada función que se encuentra en el archivo main.py, se han incluido comentarios detallados para facilitar la comprensión de la lógica detrás de su construcción y su funcionalidad. recomiendo revisar los comentarios correspondientes en el código para obtener una comprensión más profunda de la implementación de cada función.*

- ### **5. Despliegue**

    En la fase de despliegue, se utilizo el servicio en la nube utilizando Amazon Web Services (AWS). El proceso se llevó a cabo de la siguiente manera:

    1. **Creación de Instancia EC2 en AWS:**
    Se provisionó una instancia EC2 en la plataforma de AWS para alojar la aplicación. Esta instancia proporciona un entorno escalable y flexible para ejecutar la API.

    2. **Configuración del Entorno en el Servidor Remoto:**
    Se realizó una conexión SSH con el servidor remoto de Ubuntu proporcionado por Amazon. Durante esta configuración, se instaló Python y la herramienta de entornos virtuales de Python para garantizar un entorno de ejecución adecuado.

    3. **Clonación del Repositorio en Rama de Deploy:**
    Para optimizar el despliegue, se realizó una clonación del repositorio, seleccionando específicamente la rama de deploy. Esta rama contenía únicamente los elementos esenciales para el funcionamiento de la API, reduciendo así el exceso de elementos no necesarios en el entorno de producción.

    4. **Ejecución del Servidor con PM2:**
    Se utilizó PM2, una herramienta avanzada de gestión de procesos para Node.js, para ejecutar el servidor de la API. PM2 proporciona funciones como la gestión de procesos, la monitorización y la recuperación automática, asegurando un servicio continuo y sin interrupciones.


## **Uso**

Para acceder a la API, utiliza el siguiente enlace:

[Link de la API](http://18.221.69.98:8000/)

Además, para probar la API y explorar la documentación, puedes acceder a la interfaz de documentación con el siguiente enlace:

[Documentación de la API](http://18.221.69.98:8000/docs)

## **Documentación**

Puedes utilizar las funciones de la API de la siguiente manera:

- **Consulta de Información por Desarrollador:**
  - Endpoint: `/Developer/{nombre_desarrollador}`
  - Ejemplo: `/Developer/Valve`

- **Consulta de Información por Usuario:**
  - Endpoint: `/UserData/{ID_usuario}`
  - Ejemplo: `/UserData/XCOM1337`

- **Consulta de Usuario para un Género Específico:**
  - Endpoint: `/UserForGenre/{nombre_genero}`
  - Ejemplo: `/UserForGenre/Action`

- **Top 3 de Desarrolladores para un Año Dado:**
  - Endpoint: `/BestDeveloperYear/{año}`
  - Ejemplo: `/BestDeveloperYear/2015`

- **Análisis de Reseñas por Desarrolladora:**
  - Endpoint: `/DeveloperReviewsAnalysis/{nombre_desarrolladora}`
  - Ejemplo: `/DeveloperReviewsAnalysis/Valve`

- **Recomendación de Juegos por ID de Producto:**
  - Endpoint: `/Item_ItemRecommend/{ID_producto}`
  - Ejemplo: `/recomendacion_juego/10`

- **Recomendación de Juegos para un Usuario por ID:**
  - Endpoint: `/User_ItemRecommend/{ID_usuario}`
  - Ejemplo: `/User_ItemRecommend/XCOM1337`

## **Fuente de datos**

Los datos que me dieron para este proyecto fueron proporcionados a través del siguiente conjunto de datos: 

[Datasets](https://drive.google.com/drive/folders/1HqBG2-sUkz_R3h1dZU5F2uAzpRn7BSpj)

*Nota: los diccionarios de los datasets se encuentran en la carpeta img*
## **Contacto**

[<img src="https://cdn-icons-png.flaticon.com/256/174/174857.png" alt="LinkedIn" width="50"/>](https://www.linkedin.com/in/facundo-blanco-a0089024a/)   [<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/7/7e/Gmail_icon_%282020%29.svg/640px-Gmail_icon_%282020%29.svg.png" alt="Correo Electrónico" width="50"/>](mailto:blancofacundo0@gmail.com)   [<img src="https://cdn-icons-png.flaticon.com/512/25/25231.png" alt="GitHub" width="50"/>](https://github.com/FacuSB)