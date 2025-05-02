import nltk
import os
import sys

def is_resource_available(resource):
    """Verifica si un recurso específico está disponible en el entorno de Python"""
    try:
        nltk.data.find(f"tokenizers/{resource}" if resource == 'punkt' else 
                       f"taggers/{resource}" if resource == 'averaged_perceptron_tagger' else
                       f"chunkers/{resource}" if resource == 'maxent_ne_chunker' else
                       f"corpora/{resource}")
        return True
    except LookupError:
        return False

def download_nltk_resources():
    """Descarga los recursos necesarios de NLTK en el entorno de Python global"""
    print("Verificando recursos de NLTK en el entorno global de Python...")
    
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
    
    # Verificar y descargar cada recurso si es necesario
    resources_downloaded = 0
    for resource in resources:
        if is_resource_available(resource):
            print(f"Recurso '{resource}' ya está disponible en el entorno.")
            continue
            
        try:
            print(f"Descargando {resource} en el entorno global...")
            nltk.download(resource, quiet=False)
            resources_downloaded += 1
        except Exception as e:
            print(f"Error al descargar {resource}: {e}")
    
    if resources_downloaded > 0:
        print(f"Descarga de recursos NLTK completada en el entorno global de Python.")
    else:
        print("Todos los recursos NLTK ya estaban disponibles en el entorno.")
    
    print("Estos recursos estarán disponibles para todas las aplicaciones que usen este entorno de Python.")

def get_nltk_data_paths():
    """Retorna las rutas donde NLTK busca los datos"""
    return nltk.data.path

if __name__ == "__main__":
    download_nltk_resources()
    print("\nRutas de datos NLTK:")
    for path in get_nltk_data_paths():
        print(f" - {path}")
