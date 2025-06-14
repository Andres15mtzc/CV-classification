import re
import unicodedata
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
import logging
from langdetect import detect, LangDetectException

# Descargar recursos necesarios de NLTK
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    # Verificar que los recursos se descargaron correctamente
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except Exception as e:
    logging.warning(f"Error al descargar recursos de NLTK: {e}")
    # Intentar descargar de forma explícita
    nltk.download('punkt')
    nltk.download('stopwords')

# Cargar modelos de spaCy para diferentes idiomas
try:
    nlp_es = spacy.load('es_core_news_sm')
    nlp_en = spacy.load('en_core_web_sm')
    spacy_available = True
except:
    logging.warning("Modelos de spaCy no encontrados. Usando fallback simple. Para mejor rendimiento ejecute: python -m spacy download es_core_news_sm en_core_web_sm")
    spacy_available = False
    
    # Crear modelos de fallback simples
    class SimpleNLP:
        def __init__(self):
            pass
            
        def __call__(self, text):
            return SimpleDoc(text)
    
    class SimpleDoc:
        def __init__(self, text):
            self.text = text
            self.tokens = text.split()
            
        def __iter__(self):
            for token in self.tokens:
                yield SimpleToken(token)
    
    class SimpleToken:
        def __init__(self, text):
            self.text = text
            self.ent_type_ = ""
            
    # Asignar modelos de fallback
    nlp_es = SimpleNLP()
    nlp_en = SimpleNLP()

logger = logging.getLogger(__name__)

def detect_language(text):
    """Detecta el idioma del texto."""
    if not text or len(text.strip()) < 20:
        return 'en'  # Default a inglés si no hay suficiente texto
    
    try:
        return detect(text)
    except LangDetectException:
        return 'en'  # Default a inglés en caso de error

def normalize_text(text):
    """Normaliza el texto: elimina caracteres especiales y normaliza Unicode."""
    if not text:
        return ""
    
    # Normalizar Unicode
    text = unicodedata.normalize('NFKD', text)
    
    # Convertir a minúsculas
    text = text.lower()
    
    # Reemplazar URLs con token especial
    text = re.sub(r'https?://\S+|www\.\S+', ' URL ', text)
    
    # Reemplazar emails con token especial
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ' EMAIL ', text)
    
    # Reemplazar números con token especial pero mantener años y porcentajes
    text = re.sub(r'\b(?!(?:19|20)\d{2}\b)\d+%?\b', ' NUM ', text)
    
    # Mantener símbolos importantes como + # C++ .NET
    text = re.sub(r'[^\w\s+#.\-]', ' ', text)
    
    # Normalizar espacios en blanco
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def remove_bias_information(text, language='es'):
    """
    Elimina información que podría introducir sesgo en la decisión:
    - Nombres propios
    - Edades
    - Géneros
    - Referencias a nacionalidades
    - Información de contacto
    """
    # Si spaCy no está disponible, usar un enfoque basado en regex
    if not spacy_available:
        # Eliminar información de contacto (emails, teléfonos, etc.)
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
        text = re.sub(r'\b(?:\+\d{1,3}[-\s]?)?\(?\d{3}\)?[-\s]?\d{3}[-\s]?\d{4}\b', '[PHONE]', text)
        
        # Eliminar posibles nombres propios (palabras que comienzan con mayúscula)
        words = text.split()
        filtered_words = []
        for word in words:
            # Si la palabra comienza con mayúscula y no está al inicio de una oración
            if word and word[0].isupper() and len(filtered_words) > 0 and filtered_words[-1][-1] not in '.!?':
                continue
            filtered_words.append(word)
        
        return ' '.join(filtered_words)
    
    # Si spaCy está disponible, usar el enfoque basado en NER
    # Seleccionar el modelo de spaCy según el idioma
    if language.startswith('es'):
        nlp = nlp_es
    else:
        nlp = nlp_en
    
    # Procesar el texto con spaCy
    doc = nlp(text)
    
    # Eliminar entidades que podrían introducir sesgo
    bias_entities = ['PER', 'LOC', 'ORG']
    
    tokens = []
    for token in doc:
        # Mantener solo tokens que no son parte de entidades de sesgo
        if token.ent_type_ not in bias_entities:
            tokens.append(token.text)
    
    # Eliminar información de contacto (emails, teléfonos, etc.)
    text = ' '.join(tokens)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
    text = re.sub(r'\b(?:\+\d{1,3}[-\s]?)?\(?\d{3}\)?[-\s]?\d{3}[-\s]?\d{4}\b', '[PHONE]', text)
    
    return text

def tokenize_text(text, language='es'):
    """Tokeniza el texto y elimina stopwords según el idioma."""
    if not text:
        return []
    
    # Obtener stopwords para el idioma
    try:
        stop_words = set(stopwords.words(language if language != 'es' else 'spanish'))
    except:
        stop_words = set()
    
    # Tokenizar usando una implementación más simple para evitar problemas con punkt_tab
    # Dividir por espacios y eliminar puntuación
    tokens = re.findall(r'\b\w+\b', text.lower())
    
    # Eliminar stopwords
    tokens = [token for token in tokens if token.lower() not in stop_words and len(token) > 2]
    
    return tokens

def preprocess_documents(documents_dict):
    """
    Preprocesa todos los documentos en el diccionario.
    
    Args:
        documents_dict: Diccionario con ID como clave y texto como valor
        
    Returns:
        Diccionario con ID como clave y texto preprocesado como valor
    """
    processed_docs = {}
    
    for doc_id, text in documents_dict.items():
        try:
            if not text:
                processed_docs[doc_id] = {
                    'text': "",
                    'language': 'es',
                    'tokens': []
                }
                continue
                
            # Detectar idioma
            language = detect_language(text)
            
            # Normalizar texto
            normalized_text = normalize_text(text)
            
            # Eliminar información de sesgo
            unbiased_text = remove_bias_information(normalized_text, language)
            
            # Tokenizar y eliminar stopwords
            tokens = tokenize_text(unbiased_text, language)
            
            # Guardar texto procesado
            processed_docs[doc_id] = {
                'text': ' '.join(tokens),
                'language': language,
                'tokens': tokens
            }
        except Exception as e:
            logger.warning(f"Error al procesar documento {doc_id}: {str(e)}")
            # En caso de error, guardar un formato consistente
            processed_docs[doc_id] = {
                'text': str(text)[:1000] if text else "",  # Limitar longitud para evitar problemas
                'language': 'es',
                'tokens': []
            }
    
    return processed_docs
