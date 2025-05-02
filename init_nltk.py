import nltk
import os
import sys

def get_nltk_data_dir():
    """Retorna la ruta al directorio de datos NLTK del proyecto"""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nltk_data')

def is_resource_downloaded(resource, nltk_data_dir):
    """Verifica si un recurso específico ya está descargado"""
    # Diferentes recursos se almacenan en diferentes subdirectorios
    resource_dirs = {
        'punkt': 'tokenizers/punkt',
        'stopwords': 'corpora/stopwords',
        'wordnet': 'corpora/wordnet',
        'omw-1.4': 'corpora/omw-1.4',
        'averaged_perceptron_tagger': 'taggers/averaged_perceptron_tagger',
        'maxent_ne_chunker': 'chunkers/maxent_ne_chunker',
        'words': 'corpora/words'
    }
    
    # Si el recurso no está en nuestro mapeo, asumimos que no está descargado
    if resource not in resource_dirs:
        return False
    
    # Verificar si el directorio del recurso existe
    resource_path = os.path.join(nltk_data_dir, resource_dirs[resource])
    return os.path.exists(resource_path)

def download_nltk_resources():
    """Descarga los recursos necesarios de NLTK en una carpeta local del proyecto si no existen"""
    print("Verificando recursos de NLTK...")
    
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
    nltk_data_dir = get_nltk_data_dir()
    os.makedirs(nltk_data_dir, exist_ok=True)
    
    # Configurar NLTK para usar el directorio local
    nltk.data.path.insert(0, nltk_data_dir)
    
    # Verificar y descargar cada recurso si es necesario
    resources_downloaded = 0
    for resource in resources:
        if is_resource_downloaded(resource, nltk_data_dir):
            print(f"Recurso '{resource}' ya está descargado.")
            continue
            
        try:
            print(f"Descargando {resource}...")
            nltk.download(resource, download_dir=nltk_data_dir, quiet=False)
            resources_downloaded += 1
        except Exception as e:
            print(f"Error al descargar {resource}: {e}")
    
    if resources_downloaded > 0:
        print(f"Descarga de recursos NLTK completada en: {nltk_data_dir}")
    else:
        print("Todos los recursos NLTK ya estaban descargados.")
    
    print("Estos recursos estarán disponibles para todas las ejecuciones del programa.")

if __name__ == "__main__":
    download_nltk_resources()
