# Project Log: Space Biology Knowledge Engine

Este archivo sirve como un registro de la conversación y el plan de desarrollo para el motor de conocimiento de biología espacial. Su propósito es mantener el contexto y la visión del proyecto a lo largo del tiempo.

---

### **Sesión 1: Contexto Inicial y Exploración**

1.  **Objetivo del Proyecto (Definido por el Usuario):**
    *   Crear una aplicación web interactiva para organizar y resumir más de 600 publicaciones de biología espacial de la NASA.
    *   Utilizar Machine Learning (K-Means para agrupar estudios por tema y Apriori para encontrar relaciones).
    *   El objetivo es facilitar a científicos, estudiantes y planificadores de misiones la exploración de la investigación existente.

2.  **Exploración Inicial (Realizada por el Asistente):**
    *   Se verificó la estructura de archivos del proyecto, identificando las carpetas `backend`, `data` y `frontend`.
    *   Se examinaron los archivos de datos:
        *   `data/cleaned_articles.csv`: Contiene `title`, `link`, `abstract`.
        *   `data/final_dataset.csv`: Contiene las mismas columnas más `clean_abstract`, una versión del texto preprocesada y lista para el análisis.

3.  **Discusión sobre el Flujo de Trabajo de ML:**
    *   **Usuario:** Preguntó si la vectorización de texto era el siguiente paso lógico para el Machine Learning.
    *   **Asistente:** Confirmó que la vectorización es esencial. Se analizó el script `backend/src/preprocess_abstracts.py`, concluyendo que realiza una excelente limpieza de texto (minúsculas, sin stopwords, etc.) para crear la columna `clean_abstract`. Se observó que `backend/src/model_trainer.py` estaba vacío, a pesar de la existencia de modelos pre-entrenados (`.pkl`) en `backend/models`.

---

### **Sesión 2: Aclaración de Conceptos y Visión del Proyecto**

1.  **Función de los Archivos CSV y los Enlaces (Pregunta del Usuario):**
    *   **Asistente:** Se explicó que los CSVs actúan como la base de datos local del proyecto, conteniendo la metadata (título, resumen). El contenido completo de los artículos no se almacena. La columna `link` es crucial para redirigir al usuario a la publicación original y completa.

2.  **Flujo del Sistema Backend (Pregunta del Usuario):**
    *   **Asistente:** Se detalló un flujo de trabajo en dos fases:
        1.  **Fase Offline (Preparación):** Limpieza de datos, preprocesamiento de texto y entrenamiento de modelos (TF-IDF, K-Means, Apriori) que resultan en el `final_dataset.csv` y los archivos de modelo (`.pkl`, `.json`).
        2.  **Fase Online (Servicio):** La `api.py` carga los datos y modelos pre-entrenados para responder en tiempo real a las peticiones del frontend (búsquedas, visualización de clusters, etc.).

3.  **Visión del Producto Final (Discusión):**
    *   **Asistente (Propuesta Inicial):** Se propuso una interfaz visual centrada en un "mapa de clusters" interactivo.
    *   **Usuario (Corrección de Rumbo):** Rechazó la idea del mapa, prefiriendo un enfoque más tradicional y denso en información, similar a **Wikipedia**. La función principal del modelo debe ser facilitar la búsqueda y exploración por temas/clusters.
    *   **Asistente (Visión Revisada):** Se detalló una nueva visión estilo Wikipedia:
        *   Página principal con una barra de búsqueda prominente y una lista de "Temas Principales" (los clusters).
        *   Páginas de "Tema" que resumen un cluster y listan todos sus artículos.
        *   Páginas de resultados de búsqueda estándar.
        *   Páginas de artículo individual con el resumen y el enlace a la fuente original.
    *   **Usuario:** Aprobó esta nueva visión, pero indicó que los detalles de implementación se verían más adelante.

---

### **Sesión 3: Metodología de Trabajo y Próximos Pasos**

1.  **Nuevas Directivas del Usuario:**
    *   Proceder paso a paso de manera lenta y verificable.
    *   Mantener una estricta organización de archivos, creando carpetas separadas para artefactos como logs de pruebas.
    *   Crear este mismo archivo de registro para mantener el contexto.
    *   Adoptar un tono "pedante" y meticuloso.

2.  **Próximo Paso Acordado:**
    *   Revocar el plan de escribir `model_trainer.py` de una sola vez.
    *   El primer paso será crear este archivo de registro (`project_log.md`).
    *   El siguiente paso será abordar la estructura de directorios y luego, de manera incremental, empezar a construir `model_trainer.py`.

---
### **Changelog**

*   **2025-11-08 (Sesión Anterior):**
    *   `Actualizado`: `backend/src/preprocess_abstracts.py` para incluir `drop_duplicates` basado en `link`.
    *   `Ejecutado`: Script `backend/src/preprocess_abstracts.py` para generar `final_dataset.csv` limpio (572 artículos).
    *   `Ejecutado`: Script `backend/src/model_trainer.py` (con `num_clusters = 5`) para reentrenar modelos y generar `data/cluster_assignments.csv` a partir de datos limpios.

*   **2025-11-10 (Sesión Actual):**
    *   `Decisión`: Se determinó que los 5 clusters principales eran demasiado generales y se decidió aplicar una estrategia de "dividir y vencer", re-clusterizando cada uno de ellos en sub-clusters más específicos.
    *   `Análisis de Sub-Clusters`: Se analizó cada uno de los 5 clusters principales con `recluster_catchall.py` y se determinaron los `k` óptimos para la subdivisión (4, 4, 4, 4, 3).
    *   `Consolidación de Clusters Finales`: Se modificó y ejecutó `consolidate_clusters.py` para generar `data/final_cluster_assignments.csv` con 19 clusters finales.
    *   `Generación del Reporte HTML Final`: Se modificó, corrigió y ejecutó `verify_clusters.py` para generar el reporte `analysis/cluster_report.html` basado en los clusters finales.
    *   `Nomenclatura de Clusters`: Se realizó un análisis cualitativo, se propusieron y aprobaron nombres temáticos para los 19 clusters.
    *   `Almacenamiento de Nombres de Clusters`: Se creó el archivo `data/cluster_names.json` con el mapeo de IDs a nombres.
    *   `Desarrollo Inicial de la API`: Se creó la primera versión de `backend/src/api.py` con endpoints para `/clusters` y `/articles`. Se verificó su funcionamiento con la documentación interactiva de FastAPI.
    *   `Implementación de FP-Growth`:
        *   **`Modificado`:** Se añadió la lógica para el análisis de Apriori a `backend/src/model_trainer.py`.
        *   **`Fallo (Intento 1)`:** La ejecución falló con un `ArrayMemoryError` (+42 GiB), diagnosticado por un exceso de características únicas (+11,000).
        *   **`Modificado`:** Se ajustó la lógica para usar solo las 1000 características principales del TF-IDF.
        *   **`Fallo (Intento 2)`:** La re-ejecución volvió a fallar con un `ArrayMemoryError` (+23 GiB), demostrando que el algoritmo Apriori estándar de `mlxtend` no es suficientemente eficiente para este problema.
        *   **`Decisión Técnica`:** Se decidió pivotar del algoritmo Apriori al algoritmo **FP-Growth**, que logra el mismo objetivo (encontrar ítems frecuentes) pero es significativamente más eficiente en el uso de memoria.
        *   **`Modificado`:** Se actualizó `backend/src/model_trainer.py` para usar `fpgrowth` en lugar de `apriori` y se ajustó `min_support` a `0.04` para optimizar el rendimiento.
        *   **`Ejecutado con Éxito`:** El script `model_trainer.py` se ejecutó con éxito, generando `backend/models/apriori_rules.json` con 10995 reglas de asociación.
        *   **`Extensión de la API`:** Se modificó `backend/src/api.py` para cargar `apriori_rules.json` al inicio y se añadió el endpoint `GET /associations` para servir las reglas de asociación, permitiendo filtros por término y paginación.
        *   **`Validación`:** Se verificó el funcionamiento del nuevo endpoint `/associations` a través de la documentación interactiva de FastAPI, confirmando que las reglas generadas son lógicas y coherentes con el dominio de la biología espacial.

---
### **Resumen de Estado y Próximos Pasos**

**1. Estado Actual del Proyecto:**
*   **Calidad de Datos:** El dataset principal, `data/final_dataset.csv`, contiene **572 artículos únicos**.
*   **Modelo Final:** Se ha implementado una estructura de clustering jerárquica. Los 5 clusters principales se han subdividido en un total de 19 sub-clusters (4+4+4+4+3), cada uno con un ID único a partir de 100.
*   **Asignaciones Finales:** Las asignaciones de cluster definitivas y consolidadas se encuentran en `data/final_cluster_assignments.csv`.
*   **Nomenclatura:** Los 19 clusters finales tienen nombres temáticos descriptivos y aprobados, almacenados en `data/cluster_names.json`.
*   **API Backend:** La primera versión de la API (`backend/src/api.py`) está escrita, cargando todos los datos y exponiendo endpoints para clusters, artículos y reglas de asociación.
*   **Verificación de API:** Se ha verificado que la API funciona correctamente a través de la documentación interactiva de FastAPI (`/docs`).

**2. Próximos Pasos:**
*   **Desarrollo del Frontend:** Iniciar el desarrollo del frontend en React para consumir los datos de la API y construir la interfaz de usuario estilo Wikipedia.
*   **Limpieza:** Eliminar los scripts de análisis temporales (`recluster_catchall.py`, `verify_clusters.py`, `name_clusters.py`, etc.) y los gráficos del método del codo (`elbow_method_cluster_X.png`).
