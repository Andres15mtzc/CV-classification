import nltk
import os
import sys

def download_nltk_resources():
    """Descarga los recursos necesarios de NLTK"""
    print("Descargando recursos de NLTK...")
    
    # Lista de recursos a descargar
    resources = [
        'punkt',
        'stopwords',
        'wordnet',
        'omw-1.4',
        'averaged_perceptron_tagger',
        'maxent_ne_chunker',
        'words'
    ]
    
    # Crear directorio para datos de NLTK si no existe
    nltk_data_dir = os.path.expanduser('~/nltk_data')
    os.makedirs(nltk_data_dir, exist_ok=True)
    
    # Descargar cada recurso
    for resource in resources:
        try:
            print(f"Descargando {resource}...")
            nltk.download(resource, quiet=False)
        except Exception as e:
            print(f"Error al descargar {resource}: {e}")
    
    print("Descarga de recursos NLTK completada.")

if __name__ == "__main__":
    download_nltk_resources()
