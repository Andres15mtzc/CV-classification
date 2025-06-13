import os
import sys
from pathlib import Path

# Asegurar que podemos importar desde el directorio raíz
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Crear directorios necesarios
os.makedirs("data", exist_ok=True)
os.makedirs("data/jobs", exist_ok=True)
os.makedirs("data/cvs", exist_ok=True)
os.makedirs("data/results", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Importar y ejecutar la aplicación web
from src.web_ui import main

if __name__ == "__main__":
    main()
