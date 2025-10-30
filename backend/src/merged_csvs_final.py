import pandas as pd
import os

# Rutas a tus CSV
CLEANED_CSV = "../../data/cleaned_articles.csv"
GENERATED_CSV = "../../data/generated_abstracts.csv"
FINAL_CSV = "../../data/final_merged.csv"
LOG_FILE = "../../data/merge_log.txt"

# Leer CSV
df_cleaned = pd.read_csv(CLEANED_CSV)
df_generated = pd.read_csv(GENERATED_CSV)

# Marcar la fuente
df_cleaned["source"] = "cleaned"
df_generated["source"] = "generated"

# Concatenar
df_all = pd.concat([df_cleaned, df_generated], ignore_index=True)

# Detectar duplicados
dupes = df_all[df_all.duplicated(subset=["title"], keep=False)].sort_values("title")

# Crear log
with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write("=== Duplicados detectados ===\n")
    if not dupes.empty:
        for title in dupes["title"].unique():
            rows = dupes[dupes["title"] == title]
            f.write(f"\nTítulo duplicado: {title}\n")
            for idx, row in rows.iterrows():
                f.write(f"  Fuente: {row['source']}, Abstract: {row['abstract'][:60]}...\n")
            f.write("  -> Se conservará la primera ocurrencia y se descartarán las demás.\n")
    else:
        f.write("No se detectaron duplicados.\n")

# Eliminar duplicados (manteniendo la primera aparición)
df_final = df_all.drop_duplicates(subset=["title"], keep="first")

# Guardar CSV final
os.makedirs(os.path.dirname(FINAL_CSV), exist_ok=True)
df_final.to_csv(FINAL_CSV, index=False)

print(f"CSV final guardado en: {FINAL_CSV}")
print(f"Se generó log de duplicados en: {LOG_FILE}")
print(f"Total artículos finales: {len(df_final)}")
