# Sistema de Clasificación de CVs

Este sistema analiza CVs y ofertas de trabajo para determinar si un candidato debe ser aceptado o rechazado para una posición específica.

## Características

- Procesamiento de documentos en múltiples formatos (PDF, DOCX, HTML, imágenes)
- Soporte multilingüe para análisis de texto
- Análisis semántico utilizando BERT
- Extracción de palabras clave relevantes
- Modelo de clasificación XGBoost
- Métricas de rendimiento y visualizaciones

## Requisitos

```
pip install -r requirements.txt
```

Además, necesitarás instalar los modelos de spaCy:

```
python -m spacy download es_core_news_sm
python -m spacy download en_core_web_sm
```

Para el OCR, necesitarás instalar Tesseract:
- Windows: https://github.com/UB-Mannheim/tesseract/wiki
- Linux: `sudo apt-get install tesseract-ocr`
- macOS: `brew install tesseract`

## Estructura del Proyecto

- `main.py`: Punto de entrada principal
- `src/data_loader.py`: Carga de datos desde diferentes formatos
- `src/preprocessing.py`: Preprocesamiento de texto
- `src/feature_engineering.py`: Extracción de características con BERT
- `src/model.py`: Entrenamiento y evaluación del modelo XGBoost

## Uso

```
python main.py --applications ruta/a/applications.parquet --offers_dir ruta/a/ofertas --cv_dir ruta/a/cvs --output_dir resultados
```

## Resultados

El sistema genera:
- Matriz de confusión
- Informe de clasificación
- Distribución de afinidad
- Importancia de características
- Predicciones detalladas con porcentajes de afinidad
