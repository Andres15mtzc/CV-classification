import nltk
import os
import sys

def download_nltk_resources():
    """Descarga los recursos necesarios de NLTK en una carpeta local del proyecto"""
    print("Descargando recursos de NLTK en carpeta local...")
    
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
    
    # Crear directorio para datos de NLTK dentro del proyecto
    nltk_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nltk_data')
    os.makedirs(nltk_data_dir, exist_ok=True)
    
    # Configurar NLTK para usar el directorio local
    nltk.data.path.insert(0, nltk_data_dir)
    
    # Descargar cada recurso
    for resource in resources:
        try:
            print(f"Descargando {resource}...")
            nltk.download(resource, download_dir=nltk_data_dir, quiet=False)
        except Exception as e:
            print(f"Error al descargar {resource}: {e}")
    
    print(f"Descarga de recursos NLTK completada en: {nltk_data_dir}")
    print("Estos recursos estarán disponibles para todas las ejecuciones del programa.")

if __name__ == "__main__":
    download_nltk_resources()
