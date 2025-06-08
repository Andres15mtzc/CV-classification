import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from sklearn.metrics.pairwise import cosine_similarity
import logging
import re
import os
import random

# Configurar variable de entorno para evitar el error de torch.compiler
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

logger = logging.getLogger(__name__)

# Usar TF-IDF en lugar de BERT para evitar problemas de compatibilidad
class TextEncoder:
    def __init__(self, max_features=5000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=2,
            max_df=0.85,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
    def fit(self, texts):
        """Entrena el vectorizador con los textos proporcionados."""
        self.vectorizer.fit(texts)
        
    def encode(self, texts, batch_size=None):
        """Codifica textos usando TF-IDF."""
        return self.vectorizer.transform(texts).toarray()

def extract_keywords(text, language):
    """
    Extrae palabras clave relevantes del texto basadas en patrones específicos
    para habilidades técnicas, educación, experiencia, etc.
    """
    keywords = []
    
    # Patrones para habilidades técnicas (adaptados para español e inglés)
    tech_patterns = [
        r'\b(?:python|java|c\+\+|javascript|html|css|sql|nosql|mongodb|react|angular|vue|node\.js|django|flask|spring|aws|azure|gcp|docker|kubernetes|git|ci/cd|jenkins|terraform|ansible)\b',
        r'\b(?:machine learning|deep learning|nlp|computer vision|data science|big data|hadoop|spark|tableau|power bi|excel|word|powerpoint|photoshop|illustrator|indesign)\b',
        r'\b(?:aprendizaje automático|inteligencia artificial|procesamiento de lenguaje natural|visión por computadora|ciencia de datos)\b'
    ]
    
    # Patrones para educación
    edu_patterns = [
        r'\b(?:phd|ph\.d|doctor|doctorate|master|msc|mba|bachelor|licenciatura|ingenier[oa]|técnico)\b',
        r'\b(?:universidad|university|instituto|institute|college|escuela|school)\b'
    ]
    
    # Patrones para experiencia
    exp_patterns = [
        r'\b(?:\d+\s+años?|years?)\b',
        r'\b(?:experiencia|experience|senior|junior|mid|level)\b'
    ]
    
    # Combinar todos los patrones
    all_patterns = tech_patterns + edu_patterns + exp_patterns
    
    # Extraer palabras clave basadas en patrones
    for pattern in all_patterns:
        matches = re.finditer(pattern, text.lower())
        for match in matches:
            keywords.append(match.group())
    
    return list(set(keywords))  # Eliminar duplicados

def calculate_similarity(cv_vector, offer_vector):
    """Calcula la similitud coseno entre un CV y una oferta."""
    return cosine_similarity(cv_vector.reshape(1, -1), offer_vector.reshape(1, -1))[0][0]

def create_dummy_data(processed_offers, processed_cvs, num_samples=10):
    """
    Crea datos de ejemplo para pruebas cuando no hay datos reales disponibles.
    
    Args:
        processed_offers: Diccionario con ofertas procesadas
        processed_cvs: Diccionario con CVs procesados
        num_samples: Número de muestras a crear
        
    Returns:
        X: Matriz de características
        y: Vector de etiquetas
        offer_ids: Lista de IDs de ofertas
        cv_ids: Lista de IDs de CVs
    """
    logger.info("Creando datos de ejemplo para pruebas")
    
    # Usar las primeras ofertas y CVs disponibles
    offer_ids_sample = list(processed_offers.keys())[:min(num_samples, len(processed_offers))]
    cv_ids_sample = list(processed_cvs.keys())[:min(num_samples, len(processed_cvs))]
    
    # Si no hay suficientes datos, crear IDs ficticios
    if not offer_ids_sample:
        offer_ids_sample = [f"dummy_offer_{i}" for i in range(num_samples)]
    if not cv_ids_sample:
        cv_ids_sample = [f"dummy_cv_{i}" for i in range(num_samples)]
    
    # Listas para almacenar datos
    offer_texts = []
    offer_ids_list = []
    cv_texts = []
    cv_ids_list = []
    labels = []
    
    # Crear aplicaciones de ejemplo con balance de clases
    total_samples = min(len(offer_ids_sample), len(cv_ids_sample)) * 4  # 4 muestras por par (2 positivas, 2 negativas)
    
    # Asegurar que tenemos suficientes muestras
    if total_samples < 10:
        total_samples = 10
        
    # Crear muestras positivas (50%)
    positive_samples = total_samples // 2
    for i in range(positive_samples):
        offer_idx = i % len(offer_ids_sample)
        cv_idx = i % len(cv_ids_sample)
        
        offer_id = offer_ids_sample[offer_idx]
        cv_id = cv_ids_sample[cv_idx]
        
        # Extraer texto con manejo de diferentes formatos
        if offer_id in processed_offers:
            if isinstance(processed_offers[offer_id], dict) and 'text' in processed_offers[offer_id]:
                offer_text = processed_offers[offer_id]['text']
            else:
                offer_text = str(processed_offers[offer_id])
        else:
            offer_text = f"Texto de oferta de ejemplo {i}"
            
        if cv_id in processed_cvs:
            if isinstance(processed_cvs[cv_id], dict) and 'text' in processed_cvs[cv_id]:
                cv_text = processed_cvs[cv_id]['text']
            else:
                cv_text = str(processed_cvs[cv_id])
        else:
            cv_text = f"Texto de CV de ejemplo {i}"
        
        # Crear instancia con etiqueta 1 (coincide)
        offer_texts.append(offer_text)
        offer_ids_list.append(f"{offer_id}_pos_{i}")
        cv_texts.append(cv_text)
        cv_ids_list.append(f"{cv_id}_pos_{i}")
        labels.append(1)
    
    # Crear muestras negativas (50%)
    negative_samples = total_samples - positive_samples
    for i in range(negative_samples):
        # Usar diferentes combinaciones para muestras negativas
        offer_idx = i % len(offer_ids_sample)
        cv_idx = (i + len(cv_ids_sample)//2) % len(cv_ids_sample)  # Usar CV diferente
        
        offer_id = offer_ids_sample[offer_idx]
        cv_id = cv_ids_sample[cv_idx]
        
        # Extraer texto con manejo de diferentes formatos
        if offer_id in processed_offers:
            if isinstance(processed_offers[offer_id], dict) and 'text' in processed_offers[offer_id]:
                offer_text = processed_offers[offer_id]['text']
            else:
                offer_text = str(processed_offers[offer_id])
        else:
            offer_text = f"Texto de oferta de ejemplo {i}"
            
        if cv_id in processed_cvs:
            if isinstance(processed_cvs[cv_id], dict) and 'text' in processed_cvs[cv_id]:
                cv_text = processed_cvs[cv_id]['text']
            else:
                cv_text = str(processed_cvs[cv_id])
        else:
            cv_text = f"Texto de CV de ejemplo {i}"
        
        # Crear instancia con etiqueta 0 (no coincide)
        offer_texts.append(offer_text)
        offer_ids_list.append(f"{offer_id}_neg_{i}")
        cv_texts.append(cv_text)
        cv_ids_list.append(f"{cv_id}_neg_{i}")
        labels.append(0)
    
    # Crear características ficticias
    dummy_features = np.zeros((len(offer_texts), 4))  # 4 features
    for i in range(len(offer_texts)):
        # Add some random variation to avoid identical samples
        dummy_features[i, 0] = len(offer_texts[i]) / 1000  # normalized text length
        dummy_features[i, 1] = len(cv_texts[i]) / 1000  # normalized text length
        dummy_features[i, 2] = np.random.random() * 0.5  # random similarity
        dummy_features[i, 3] = np.random.randint(0, 5)  # random keyword matches
    
    return dummy_features, np.array(labels), offer_ids_list, cv_ids_list

def extract_features(applications_df, processed_offers, processed_cvs):
    """
    Extrae características para el modelo de clasificación.
    
    Args:
        applications_df: DataFrame con las relaciones entre ofertas y aplicantes
        processed_offers: Diccionario con ofertas procesadas
        processed_cvs: Diccionario con CVs procesados
        
    Returns:
        X: Matriz de características
        y: Vector de etiquetas (aceptado/rechazado)
        offer_ids: IDs de ofertas correspondientes a cada fila
        cv_ids: IDs de CVs correspondientes a cada fila
    """
    # Inicializar el codificador de texto (TF-IDF en lugar de BERT)
    text_encoder = TextEncoder(max_features=1000)
    
    # Preparar datos para vectorización
    offer_texts = []
    offer_ids_list = []
    cv_texts = []
    cv_ids_list = []
    labels = []
    
    # Extraer textos y etiquetas de las aplicaciones
    # Verificar las columnas disponibles en el DataFrame
    logger.info(f"Columnas disponibles en applications_df: {applications_df.columns.tolist()}")
    
    # Determinar los nombres de las columnas para offer_id y cv_id
    offer_id_col = next((col for col in ['offer_id', 'job_id', 'oferta_id'] if col in applications_df.columns), None)
    cv_id_col = next((col for col in ['cv_id', 'applicant_id', 'candidato_id'] if col in applications_df.columns), None)
    
    if not offer_id_col or not cv_id_col:
        logger.error(f"No se encontraron columnas para offer_id o cv_id en el DataFrame")
        # Usar la función auxiliar para crear datos de ejemplo
        return create_dummy_data(processed_offers, processed_cvs)
    
    # Si encontramos las columnas correctas, procesamos normalmente
    for _, row in applications_df.iterrows():
        offer_id = str(row[offer_id_col])
        cv_id = str(row[cv_id_col])
        
        # Verificar si tenemos los textos procesados
        if offer_id not in processed_offers or cv_id not in processed_cvs:
            logger.warning(f"Falta texto procesado para oferta {offer_id} o CV {cv_id}")
            continue
        
        # Verificar el tipo de datos y extraer el texto
        if isinstance(processed_offers[offer_id], dict) and 'text' in processed_offers[offer_id]:
            offer_text = processed_offers[offer_id]['text']
        else:
            offer_text = str(processed_offers[offer_id])
            
        if isinstance(processed_cvs[cv_id], dict) and 'text' in processed_cvs[cv_id]:
            cv_text = processed_cvs[cv_id]['text']
        else:
            cv_text = str(processed_cvs[cv_id])
        
        # Agregar a las listas
        offer_texts.append(offer_text)
        offer_ids_list.append(offer_id)
        cv_texts.append(cv_text)
        cv_ids_list.append(cv_id)
        
        # Etiqueta (si está disponible en el DataFrame)
        if 'accepted' in row:
            labels.append(int(row['accepted']))
        else:
            # Si no hay etiqueta, asumimos que estamos en modo de inferencia
            labels.append(-1)
    
    # Verificar si tenemos datos para procesar
    if not offer_texts or not cv_texts:
        logger.error("No hay datos para procesar. Creando datos de ejemplo.")
        # Usar la función auxiliar para crear datos de ejemplo
        return create_dummy_data(processed_offers, processed_cvs)
    
    # Vectorizar textos con TF-IDF
    logger.info("Entrenando vectorizador TF-IDF...")
    all_texts = offer_texts + cv_texts
    text_encoder.fit(all_texts)
    
    logger.info("Codificando ofertas con TF-IDF...")
    offer_vectors = text_encoder.encode(offer_texts)
    
    logger.info("Codificando CVs con TF-IDF...")
    cv_vectors = text_encoder.encode(cv_texts)
    
    # Calcular similitud entre ofertas y CVs
    similarities = []
    for i in range(len(offer_vectors)):
        similarity = calculate_similarity(cv_vectors[i], offer_vectors[i])
        similarities.append(similarity)
    
    # Extraer palabras clave
    cv_keywords = []
    offer_keywords = []
    keyword_matches = []
    
    for i in range(len(offer_texts)):
        offer_id = offer_ids_list[i]
        cv_id = cv_ids_list[i]
        
        # Obtener idioma con manejo de diferentes tipos de datos
        if isinstance(processed_offers[offer_id], dict) and 'language' in processed_offers[offer_id]:
            offer_lang = processed_offers[offer_id]['language']
        else:
            offer_lang = 'es'  # Idioma por defecto
            
        if isinstance(processed_cvs[cv_id], dict) and 'language' in processed_cvs[cv_id]:
            cv_lang = processed_cvs[cv_id]['language']
        else:
            cv_lang = 'es'  # Idioma por defecto
        
        # Extraer palabras clave
        offer_kw = extract_keywords(offer_texts[i], offer_lang)
        cv_kw = extract_keywords(cv_texts[i], cv_lang)
        
        # Calcular coincidencias de palabras clave
        matches = len(set(offer_kw).intersection(set(cv_kw)))
        
        offer_keywords.append(offer_kw)
        cv_keywords.append(cv_kw)
        keyword_matches.append(matches)
    
    # Crear DataFrame con características
    features_df = pd.DataFrame({
        'offer_id': offer_ids_list,
        'cv_id': cv_ids_list,
        'bert_similarity': similarities,
        'keyword_matches': keyword_matches,
        'offer_length': [len(text.split()) for text in offer_texts],
        'cv_length': [len(text.split()) for text in cv_texts],
    })
    
    # Agregar características adicionales si es necesario
    
    # Imprimir información sobre el DataFrame de características
    logger.info(f"DataFrame de características creado con {len(features_df)} filas")
    logger.info(f"Columnas del DataFrame: {features_df.columns.tolist()}")
    logger.info(f"Primeras 5 filas del DataFrame:\n{features_df.head()}")
    
    # Imprimir información sobre las etiquetas
    unique_labels = set(labels)
    label_counts = {label: labels.count(label) for label in unique_labels}
    logger.info(f"Etiquetas únicas y sus conteos: {label_counts}")
    
    # Convertir a matriz numpy para el modelo
    try:
        X = features_df.drop(['offer_id', 'cv_id'], axis=1).values
        y = np.array(labels)
        
        logger.info(f"Matriz X creada con forma {X.shape}")
        logger.info(f"Vector y creado con forma {y.shape}")
        
        # Verificar que X es una matriz numérica 2D
        if not isinstance(X, np.ndarray) or X.ndim != 2 or not np.issubdtype(X.dtype, np.number):
            logger.warning("X no es una matriz numérica 2D válida. Creando características ficticias.")
            # Crear características ficticias
            X = np.zeros((len(offer_ids_list), 4))
            for i in range(len(offer_ids_list)):
                X[i, 0] = len(offer_texts[i]) / 1000 if i < len(offer_texts) else 0.1
                X[i, 1] = len(cv_texts[i]) / 1000 if i < len(cv_texts) else 0.1
                X[i, 2] = similarities[i] if i < len(similarities) else 0.5
                X[i, 3] = keyword_matches[i] if i < len(keyword_matches) else 2
    except Exception as e:
        logger.error(f"Error al crear matriz de características: {e}")
        # Crear características ficticias en caso de error
        X = np.zeros((len(offer_ids_list), 4))
        for i in range(len(offer_ids_list)):
            X[i, 0] = len(offer_texts[i]) / 1000 if i < len(offer_texts) else 0.1
            X[i, 1] = len(cv_texts[i]) / 1000 if i < len(cv_texts) else 0.1
            X[i, 2] = np.random.random() * 0.5
            X[i, 3] = np.random.randint(0, 5)
    
    return X, y, offer_ids_list, cv_ids_list
