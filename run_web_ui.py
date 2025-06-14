import os
import sys
import logging
import shutil
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

# Importar funciones para crear plantillas
from src.web_ui import create_templates, create_template_directories

# Forzar la recreación de plantillas
logger.info("Creando plantillas...")
templates_dir, static_dir = create_template_directories()
create_templates()
logger.info(f"Plantillas creadas en: {templates_dir}")

# Verificar que las plantillas existen
template_files = os.listdir(templates_dir)
logger.info(f"Archivos de plantillas: {template_files}")

if not template_files or "base.html" not in template_files:
    logger.error("¡Error! Las plantillas no se crearon correctamente.")
    sys.exit(1)

# Importar y ejecutar la aplicación web
from src.web_ui import main

if __name__ == "__main__":
    try:
        logger.info("Iniciando aplicación web...")
        print("\n=== CVQ - CV's Qualification Web UI ===")
        print("Abre tu navegador en: http://127.0.0.1:5000")
        print("Presiona Ctrl+C para detener el servidor\n")
        main()
    except Exception as e:
        logger.error(f"Error al iniciar la aplicación: {str(e)}")
