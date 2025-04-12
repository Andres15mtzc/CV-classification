"""
Script para inicializar y descargar todos los recursos necesarios
para el sistema de clasificación de CVs.
"""

import nltk
import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Descargando recursos de NLTK...")
    nltk.download('punkt')
    nltk.download('stopwords')
    
    logger.info("Instalando modelos de spaCy...")
    os.system("python -m spacy download es_core_news_sm")
    os.system("python -m spacy download en_core_web_sm")
    
    logger.info("Verificando que los recursos se hayan descargado correctamente...")
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        logger.info("Recursos de NLTK verificados correctamente.")
    except LookupError as e:
        logger.error(f"Error al verificar recursos de NLTK: {e}")
        sys.exit(1)
    
    logger.info("Todos los recursos han sido descargados correctamente.")

if __name__ == "__main__":
    main()
