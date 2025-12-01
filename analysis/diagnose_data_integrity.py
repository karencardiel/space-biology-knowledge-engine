import pandas as pd
import os

# === RUTAS ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FINAL_DATASET_PATH = os.path.join(BASE_DIR, "../data/final_dataset.csv")
INITIAL_ASSIGNMENTS_PATH = os.path.join(BASE_DIR, "../data/cluster_assignments.csv")
FINAL_ASSIGNMENTS_PATH = os.path.join(BASE_DIR, "../data/final_cluster_assignments.csv")

def analyze_file(path, name):
    print(f"--- Analizando Archivo: {name} ---")
    if not os.path.exists(path):
        print("Archivo no encontrado.\n")
        return None
    
    df = pd.read_csv(path)
    total_rows = len(df)
    unique_links = df['link'].nunique()
    num_duplicates = total_rows - unique_links
    
    print(f"  - Filas totales: {total_rows}")
    print(f"  - Links únicos: {unique_links}")
    print(f"  - Filas con links duplicados: {num_duplicates}\n")
    return df

# --- Ejecutar Análisis ---
df_final_dataset = analyze_file(FINAL_DATASET_PATH, "final_dataset.csv")
df_initial_assignments = analyze_file(INITIAL_ASSIGNMENTS_PATH, "cluster_assignments.csv")
df_final_assignments = analyze_file(FINAL_ASSIGNMENTS_PATH, "final_cluster_assignments.csv")

# --- Simular la Unión ---
if df_final_dataset is not None and df_final_assignments is not None:
    print("--- Simulando la Unión (Merge) ---")
    # Asegurémonos de que el df principal no tenga duplicados para una prueba limpia
    df_main_clean = df_final_dataset.drop_duplicates(subset=['link'])
    
    merged_df = pd.merge(df_main_clean, df_final_assignments, on='link', how='left')
    print(f"Filas en final_dataset.csv (limpio): {len(df_main_clean)}")
    print(f"Filas en final_cluster_assignments.csv: {len(df_final_assignments)}")
    print(f"Filas después de la unión: {len(merged_df)}\n")
    
    if len(merged_df) > len(df_final_assignments):
        print("¡ALERTA! La operación de unión está creando filas duplicadas. Esto sugiere que hay `links` en `final_cluster_assignments.csv` que no están en `final_dataset.csv` o viceversa, causando problemas en el `merge`.")
    elif len(merged_df) > len(df_main_clean):
         print("¡ALERTA! La operación de unión está creando filas duplicadas. Esto confirma que hay `links` duplicados en `final_cluster_assignments.csv`.")
    else:
        print("La operación de unión parece ser estable y no crea filas adicionales.")
