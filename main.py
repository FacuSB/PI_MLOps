from fastapi import FastAPI
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from fastapi.responses import JSONResponse

####################################################################################################################
#Dataframe Original
df_original = pd.read_parquet('Data/final.parquet.gzip')

#Dataframe Steam_Games
df_steam_games=pd.read_json('Data/steam_games.json',lines=True)
df_steam_games=df_steam_games.drop(columns=['Unnamed: 0'])

#Dataframe Review
df_review = df_original.loc[:, ['user_id','recomended_item_id','sentiment_analysis','recommend']]
df_review=df_review.dropna()
df_review['recomended_item_id'] = df_review['recomended_item_id'].apply(lambda x: [int(i) for i in x])
    # Primero, eliminamos los duplicados en df_steam_games
df_steam_games = df_steam_games.drop_duplicates(subset='id')

    # Convertimos df_steam_games a un diccionario para facilitar la búsqueda
dict_df2 = df_steam_games.set_index('id').to_dict('index')

    # Creamos una nueva columna en df1 que contendrá los datos correspondientes de df2
df_review['games_info'] = df_review['recomended_item_id'].apply(lambda x: [dict_df2.get(i, {}) for i in x])


#Dataframe Usuarios
df_user_info=df_original.copy()

#Dataframe para funcion UserForGenre
df_user_with_most_playtime=pd.read_json('data_optimizada/user_with_most_playtime.json',lines=True)
####################################################################################################################






################################################ FUNCION 1 ################################################
def Developer_Funct(developer: str):
    # Hacer una copia de los dataframes
    df_steam_games_copy = df_steam_games.copy()

    # Filtrar el dataframe por el desarrollador
    dev_games = df_steam_games_copy[df_steam_games_copy['developer'] == developer]

    # Calcular la cantidad de items por año
    dev_games['release_year'] = pd.to_datetime(dev_games['release_date']).dt.year
    items_per_year = dev_games.groupby('release_year').size().to_dict()

    # Calcular el porcentaje de contenido gratuito por año
    free_content = dev_games[dev_games['price'] == 0]
    free_content_per_year = (free_content.groupby('release_year').size() / dev_games.groupby('release_year').size() * 100).fillna(0)
    # Redondear el porcentaje y darle formato"
    free_content_per_year = {year: f"{round(percentage)}%" for year, percentage in free_content_per_year.items()}

    # Convertir los resultados a formato JSON
    result = {
        'developer': developer,
        'items_per_year': items_per_year,
        'free_content_per_year': free_content_per_year
    }

    return result

################################################ FUNCION 2 ################################################

def UserData_Funct(User_id: str):
    # Hacer una copia de los dataframes
    df_user_info_copy = df_user_info.copy()
    df_steam_games_copy = df_steam_games.copy()
    df_review_copy = df_review.copy()

    # Filtrar el dataframe df_user_info_copy por el User_id
    user_data = df_user_info_copy[df_user_info_copy['user_id'] == User_id]

    # Comprobar si user_data está vacío
    if not user_data.empty:
        # Obtener la lista de ids de items del usuario
        user_items = user_data['user_item_id'].iloc[0]
    else:
        user_items = []

    # Filtrar el dataframe df_steam_games_copy por los ids de items del usuario
    user_games = df_steam_games_copy[df_steam_games_copy['id'].isin(user_items)]

    # Calcular la cantidad de dinero gastado por el usuario y redondearlo a dos decimales
    money_spent = round(float(user_games['price'].sum()), 2)

    # Filtrar el dataframe df_review_copy por el User_id
    user_reviews = df_review_copy[df_review_copy['user_id'] == User_id]

    # Comprobar si user_reviews está vacío
    if not user_reviews.empty:
        # Obtener la lista de recomendaciones del usuario
        recommendations = user_reviews['recommend'].iloc[0]
    else:
        recommendations = []

    # Calcular el porcentaje de recomendación
    recommend_percentage = round((sum(recommendations) / len(user_items)) * 100, 2) if user_items is not None and len(user_items) > 0 else 0

    # Calcular la cantidad de items y la cantidad de items recomendados
    num_items = len(user_items)
    num_recommended_items = sum(recommendations)

    # Crear resultado
    result = {
        "Usuario": User_id,
        "Dinero gastado": float(f"{money_spent}"),
        "cantidad de items": int(num_items),
        "cantidad de items recomendados": int(num_recommended_items),
        "% de recomendación": float(f"{recommend_percentage}")
    }

    return result

################################################ FUNCION 3 ################################################
def UserForGenre_Funct(genre: str):
    # Filtrar el DataFrame por el género dado
    df_genre = df_user_with_most_playtime[df_user_with_most_playtime['Genero'] == genre]

    # Si no hay datos para el género dado, devolver un mensaje indicando esto
    if df_genre.empty:
        return f"No hay datos disponibles para el género {genre}."

    # Obtener el usuario con más horas jugadas para el género dado
    user = df_genre['Usuario con más horas jugadas'].iloc[0]

    # Obtener la lista de la acumulación de horas jugadas por año
    playtime_by_year = df_genre['Años y Horas'].iloc[0]

    # Crear resultado
    return {"Usuario con más horas jugadas para " + genre: user, "Horas jugadas": playtime_by_year}

################################################ FUNCION 4 ################################################
def BestDeveloperYear_Funct(año: int):
    # Hacer una copia de los dataframes
    df_review_copy = df_review.copy()
    df_steam_games_copy = df_steam_games.copy()

    # Asegúrate de tener una columna 'release_date' en formato datetime
    df_steam_games_copy['release_date'] = pd.to_datetime(df_steam_games_copy['release_date'])
    
    # Filtrar los juegos por año
    df_games_year = df_steam_games_copy[df_steam_games_copy['release_date'].dt.year == año]

    # Crear un diccionario para contar las reseñas por desarrollador
    developer_counts = {}

    # Iterar sobre cada fila en df_review
    for index, row in df_review_copy.iterrows():
    # Iterar sobre los pares de game_id y recommend en las listas 'recomended_item_id' y 'recommend'
        for game_id, recommend in zip(row['recomended_item_id'], row['recommend']):
            # Si el juego fue recomendado
            if recommend:
                game_row = df_games_year[df_games_year['id'] == game_id]
                # Si el juego existe en df_games_year (es decir, fue lanzado en el año especificado)
                if not game_row.empty:
                    # Obtener el nombre del desarrollador del juego
                    developer = game_row.iloc[0]['developer']
                    if developer in developer_counts:
                    # Si el desarrollador ya está en el diccionario developer_counts, incrementar su conteo
                        developer_counts[developer] += 1
                    else:
                    # Si el desarrollador no está en el diccionario developer_counts, lo añade y pone su contador en 1
                        developer_counts[developer] = 1

    # Ordenar los desarrolladores por cantidad de reseñas y tomar los primeros 3
    top_3_developers = sorted(developer_counts.items(), key=lambda x: x[1], reverse=True)[:3]

    # Crear resultado
    result = [{"Puesto {}".format(i+1): developer[0]} for i, developer in enumerate(top_3_developers)]

    return result

################################################ FUNCION 5 ################################################
def DeveloperReviewsAnalysis_Funct(desarrolladora: str):
    # Hacer una copia de los dataframes
    df_review_copy = df_review.copy()
    df_steam_games_copy = df_steam_games.copy()

    # Crear un diccionario para contar las reseñas por desarrollador
    developer_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}

    # Iterar sobre cada fila en df_review
    for index, row in df_review_copy.iterrows():
        for game_id, recommend, sentiment in zip(row['recomended_item_id'], row['recommend'], row['sentiment_analysis']):
            if recommend:
                game_row = df_steam_games_copy[df_steam_games_copy['id'] == game_id]
                if not game_row.empty and game_row.iloc[0]['developer'] == desarrolladora:
                    # Convertir los valores numéricos a cadenas
                    if sentiment == 2:
                        sentiment = "Positive"
                    elif sentiment == 0:
                        sentiment = "Negative"
                    elif sentiment == 1:
                        sentiment = "Neutral"
                    developer_counts[sentiment] += 1

    # Crear resultado
    result = {desarrolladora: [f"Negative = {developer_counts['Negative']}", f"Neutral = {developer_counts['Neutral']}", f"Positive = {developer_counts['Positive']}"]}

    return result

################################################ FUNCION 6 ################################################
def Items_Recommend_Funct( id_producto: int):
    
    df=df_steam_games
    def obtener_datos_producto(df, id_producto):
        # Esta función debería devolver los datos del producto con el id dado
        producto_df = df[df['id'] == id_producto]
        if not producto_df.empty:
            producto_info = producto_df[['publisher', 'genres', 'app_name', 'title', 'tags', 'specs', 'price', 'early_access', 'developer']].values[0]
            return ' '.join(map(str, producto_info))
        else:
            return None

    def calcular_similitud_coseno(producto, todos_los_productos):
        # Esta función debería calcular la similitud del coseno entre el producto dado y todos los demás productos
        vectorizer = TfidfVectorizer()
        producto_vec = vectorizer.fit_transform([producto])
        todos_los_productos_vec = vectorizer.transform(todos_los_productos)
        
        # Luego, calculamos la similitud del coseno
        similitudes = cosine_similarity(producto_vec, todos_los_productos_vec)
        return similitudes

    def ordenar_por_similitud(similitudes):
        # Esta función debería ordenar los productos por similitud
        return similitudes.argsort()

    # Obtén los datos del producto
    producto = obtener_datos_producto(df, id_producto)
    
    # Si el producto no existe, devuelve un mensaje de error
    if producto is None:
        return {"Error": f"No se encontró ningún producto con el id {id_producto}"}
    
    # Si el producto existe, calcula la similitud del coseno entre el producto y todos los demás productos
    todos_los_productos = [' '.join(map(str, x)) for x in df[['publisher', 'genres', 'app_name', 'title', 'tags', 'specs', 'price', 'early_access', 'developer']].values]
    similitudes = calcular_similitud_coseno(producto, todos_los_productos)
    
    # Ordena los productos por similitud y toma los 5 más similares
    productos_similares = ordenar_por_similitud(similitudes)[0][-5:]
    
    # Devuelve los productos similares en formato de diccionario con id y nombre del producto
    return {id: nombre for id, nombre in zip(df.iloc[productos_similares]['id'], df.iloc[productos_similares]['app_name'])}

################################################ FUNCION 7 ################################################
def Users_Recommend_Funct( id_usuario: str):

    df_user = df_user_info
    df_item = df_steam_games

    def obtener_datos_usuario(df, id_usuario):
        # Esta función debería devolver los datos del usuario con el id dado
        usuario_df = df[df['user_id'] == id_usuario]
        if not usuario_df.empty:
            usuario_info = usuario_df[['steam_id', 'user_url', 'user_item_id', 'played_item_name', 'playtime_forever', 'playtime_2weeks', 'review_content', 'recomended_item_id', 'sentiment_analysis', 'recommend']].values[0]
            return ' '.join(map(str, usuario_info))
        else:
            return None

    def calcular_similitud_coseno(usuario, todos_los_productos):
        # Esta función debería calcular la similitud del coseno entre el usuario dado y todos los productos
        vectorizer = TfidfVectorizer()
        usuario_vec = vectorizer.fit_transform([usuario])
        todos_los_productos_vec = vectorizer.transform(todos_los_productos)
        
        # Luego, calculamos la similitud del coseno
        similitudes = cosine_similarity(usuario_vec, todos_los_productos_vec)
        return similitudes

    def ordenar_por_similitud(similitudes):
        # Esta función debería ordenar los productos por similitud
        return similitudes.argsort()

    # Obtén los datos del usuario
    usuario = obtener_datos_usuario(df_user, id_usuario)
    
    # Si el usuario no existe, devuelve un mensaje de error
    if usuario is None:
        return {"Error": f"No se encontró ningún usuario con el id {id_usuario}"}
    
    # Si el usuario existe, calcula la similitud del coseno entre el usuario y todos los productos
    todos_los_productos = [' '.join(map(str, x)) for x in df_item[['publisher', 'genres', 'app_name', 'title', 'tags', 'specs', 'price', 'early_access', 'developer']].values]
    similitudes = calcular_similitud_coseno(usuario, todos_los_productos)
    
    # Ordena los productos por similitud y toma los 5 más similares
    productos_similares = ordenar_por_similitud(similitudes)[0][-5:]
    
    # Devuelve los productos similares en formato de diccionario con id y nombre del producto
    return {id: nombre for id, nombre in zip(df_item.iloc[productos_similares]['id'], df_item.iloc[productos_similares]['app_name'])}

##########################################FastAPI######################################################################
app = FastAPI()

@app.get("/")
def index_html():
    return JSONResponse(content={"message": "¡Bienvenido a la API!"})

@app.get('/Developer/{Developer}')
def Developer(Desarrollador: str):
    
    try:
        return Developer_Funct(Desarrollador)
    except Exception as e:
        return {"Error":str(e)}
    
@app.get('/UserData/{id}')
def UserData(id: str):
    
    try:
        return UserData_Funct(id)
    except Exception as e:
        return {"Error":str(e)}
    
@app.get('/UserForGenre/{genero}')
def UserForGenre(genero: str):

    try:
        return UserForGenre_Funct(genero)
    except Exception as e:
        return {"Error":str(e)}
    
@app.get('/BestDeveloperYear/{anio}')   
def BestDeveloperYear(anio: int):
    
    try:
        return BestDeveloperYear_Funct(anio)
    except Exception as e:
        return {"Error":str(e)}
    
@app.get('/DeveloperReviewsAnalysis/{desarrolladora}') 
def DeveloperReviewsAnalysis(desarrolladora: str):
    
    try:
        return DeveloperReviewsAnalysis_Funct(desarrolladora)
    except Exception as e:
        return {"Error":str(e)}

@app.get('/Item_ItemRecommend/{id}') 
def Items_Recommend(id: int):
    
    try:
        return Items_Recommend_Funct(id)
    except Exception as e:
        return {"Error":str(e)}

@app.get('/User_ItemRecommend/{id}') 
def Users_Recommend(id: str):
    
    try:
        return Users_Recommend_Funct(id)
    except Exception as e:
        return {"Error":str(e)}