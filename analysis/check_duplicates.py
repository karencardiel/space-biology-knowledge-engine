import pandas as pd
import os

# Ruta al dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../data/final_dataset.csv")

# Cargar datos
print(f"Cargando dataset desde: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

# Contar duplicados basados en la columna 'link'
duplicate_links_mask = df.duplicated(subset=['link'], keep=False)
num_duplicates_by_link = duplicate_links_mask.sum()

print(f"\nNúmero total de filas: {len(df)}")
print(f"Número de filas con links duplicados: {num_duplicates_by_link}")

if num_duplicates_by_link > 0:
    print("\n--- Verificando contenido de filas con links duplicados ---")
    
    # Obtener solo las filas que tienen links duplicados
    df_duplicates = df[duplicate_links_mask].sort_values(by='link')
    
    # Agrupar por link y verificar si los títulos o abstracts son diferentes
    has_content_discrepancy = False
    for link, group in df_duplicates.groupby('link'):
        if group['title'].nunique() > 1 or group['abstract'].nunique() > 1:
            print(f"\n¡ADVERTENCIA! Link '{link}' tiene títulos o abstracts diferentes:")
            print(group[['title', 'abstract']])
            has_content_discrepancy = True
        # else:
            # print(f"\nLink '{link}' tiene contenido idéntico (título y abstract).") # Descomentar para ver los que son idénticos
            
    if not has_content_discrepancy:
        print("\nTodos los links duplicados tienen títulos y abstracts idénticos. Es seguro eliminar duplicados por 'link'.")
    else:
        print("\nSe encontraron links duplicados con contenido diferente. Se requiere una revisión manual.")
else:
    print("\nNo se encontraron links duplicados.")