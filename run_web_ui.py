import os
import sys
import logging
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Asegurar que podemos importar desde el directorio raíz
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Crear directorios necesarios
os.makedirs("data", exist_ok=True)
os.makedirs("data/jobs", exist_ok=True)
os.makedirs("data/cvs", exist_ok=True)
os.makedirs("data/results", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)
os.makedirs("static/css", exist_ok=True)
os.makedirs("static/js", exist_ok=True)

logger.info("Directorios creados correctamente")

# Verificar si las plantillas existen
if not os.path.exists("templates/base.html"):
    logger.info("Plantillas no encontradas, se crearán automáticamente")
    from src.web_ui import create_templates, create_template_directories
    create_template_directories()
    create_templates()
    logger.info("Plantillas creadas correctamente")
else:
    logger.info("Plantillas encontradas")

# Importar y ejecutar la aplicación web
from src.web_ui import main

if __name__ == "__main__":
    try:
        logger.info("Iniciando aplicación web...")
        main()
    except Exception as e:
        logger.error(f"Error al iniciar la aplicación: {str(e)}")
