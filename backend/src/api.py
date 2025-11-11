import os
import json
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional

# --- 1. Carga y Preparación de Datos ---

# Definir rutas absolutas para asegurar que el script funcione desde cualquier lugar
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, '../data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

print("Iniciando API: Cargando y fusionando datos...")

try:
    # Cargar datasets de artículos y clusters
    articles_df = pd.read_csv(os.path.join(DATA_DIR, 'final_dataset.csv'))
    assignments_df = pd.read_csv(os.path.join(DATA_DIR, 'final_cluster_assignments.csv'))
    with open(os.path.join(DATA_DIR, 'cluster_names.json'), 'r', encoding='utf-8') as f:
        cluster_names = json.load(f)
    
    # Fusionar dataframes de artículos
    merged_df = pd.merge(articles_df, assignments_df, on='link', how='left')
    merged_df['final_cluster'] = merged_df['final_cluster'].astype(str)
    merged_df['cluster_name'] = merged_df['final_cluster'].map(cluster_names)
    merged_df['cluster_name'] = merged_df['cluster_name'].fillna('Sin categoría')
    merged_df['final_cluster'] = merged_df['final_cluster'].fillna('-1')
    print(f"Datos de artículos cargados. Total: {len(merged_df)} artículos.")

    # Cargar las reglas de asociación
    rules_path = os.path.join(MODEL_DIR, 'apriori_rules.json')
    rules_df = pd.read_json(rules_path, orient='records')
    print(f"Reglas de asociación cargadas. Total: {len(rules_df)} reglas.")

except FileNotFoundError as e:
    print(f"Error: No se pudo encontrar un archivo de datos esencial: {e}. Asegúrese de que los archivos de datos existen.")
    exit()


# --- 2. Inicialización de la App FastAPI ---

app = FastAPI(
    title="Space Biology Knowledge Engine API",
    description="API para explorar publicaciones de biología espacial de la NASA clasificadas por temas y sus asociaciones.",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- 3. Endpoints de la API ---

@app.get("/clusters", summary="Obtener la lista de todos los clusters temáticos")
def get_clusters():
    """
    Devuelve una lista de todos los clusters temáticos, incluyendo su ID, nombre y el número de artículos que contiene cada uno.
    """
    cluster_counts = merged_df['final_cluster'].value_counts().to_dict()
    
    cluster_list = []
    for cluster_id, cluster_name in cluster_names.items():
        cluster_list.append({
            "id": cluster_id,
            "name": cluster_name,
            "article_count": cluster_counts.get(cluster_id, 0)
        })
        
    cluster_list = sorted(cluster_list, key=lambda x: int(x['id']))
    return {"clusters": cluster_list}


@app.get("/articles", summary="Obtener una lista de artículos con filtros")
def get_articles(
    cluster_id: Optional[str] = Query(None, description="Filtrar artículos por el ID del cluster temático."),
    search: Optional[str] = Query(None, description="Buscar un término en el título y el resumen de los artículos."),
    skip: int = Query(0, ge=0, description="Número de artículos a omitir (para paginación)."),
    limit: int = Query(20, ge=1, le=100, description="Número máximo de artículos a devolver.")
):
    """
    Devuelve una lista de artículos. Permite filtrar por cluster, buscar por texto y paginar los resultados.
    """
    temp_df = merged_df.copy()
    
    if search:
        search_term = search.lower()
        temp_df = temp_df[
            temp_df['title'].str.lower().str.contains(search_term) |
            temp_df['abstract'].str.lower().str.contains(search_term)
        ]
        
    if cluster_id:
        temp_df = temp_df[temp_df['final_cluster'] == cluster_id]
        
    total_results = len(temp_df)
    paginated_df = temp_df.iloc[skip : skip + limit]
    articles = paginated_df.to_dict(orient='records')
    
    return {
        "total_results": total_results,
        "articles": articles,
        "skip": skip,
        "limit": limit
    }

@app.get("/associations", summary="Explorar reglas de asociación entre palabras clave")
def get_associations(
    term: Optional[str] = Query(None, description="Término para buscar en los antecedentes de una regla."),
    min_confidence: float = Query(0.5, ge=0, le=1, description="Confianza mínima de la regla."),
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100)
):
    """
    Devuelve una lista de reglas de asociación.
    - Si se proporciona un 'term', filtra las reglas donde el término está en los antecedentes.
    - Filtra por una confianza mínima.
    - Devuelve las reglas más fuertes (mayor 'lift' y 'confidence') primero.
    """
    temp_df = rules_df[rules_df['confidence'] >= min_confidence].copy()
    
    if term:
        # Filtrar donde el término de búsqueda está en la lista de antecedentes
        temp_df = temp_df[temp_df['antecedents'].apply(lambda x: term in x)]
        
    # Ordenar por las métricas más relevantes
    temp_df = temp_df.sort_values(by=['lift', 'confidence'], ascending=False)
    
    total_results = len(temp_df)
    
    # Aplicar paginación
    paginated_df = temp_df.iloc[skip : skip + limit]
    
    rules = paginated_df.to_dict(orient='records')
    
    return {
        "total_results": total_results,
        "rules": rules,
        "skip": skip,
        "limit": limit
    }

@app.get("/", summary="Endpoint de bienvenida")
def read_root():
    return {"message": "Bienvenido a la API del Space Biology Knowledge Engine. Visite /docs para la documentación."}

print("API lista y actualizada con reglas de asociación. Visite /docs para ver la documentación interactiva.")