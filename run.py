import os
import sys
import logging
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    print("\n=== CV Matcher ===")
    print("\nSelecciona la interfaz que deseas utilizar:")
    print("1. Interfaz Gráfica (GUI)")
    print("2. Interfaz Web")
    print("3. Salir")
    
    choice = input("\nIngresa tu elección (1-3): ")
    
    if choice == "1":
        print("\nIniciando Interfaz Gráfica...")
        os.system("python run_gui.py")
    elif choice == "2":
        print("\nIniciando Interfaz Web...")
        print("Una vez iniciado, abre tu navegador en: http://127.0.0.1:5000")
        os.system("python run_web_ui.py")
    elif choice == "3":
        print("\nSaliendo...")
        sys.exit(0)
    else:
        print("\nOpción no válida. Por favor, intenta de nuevo.")
        main()

if __name__ == "__main__":
    # Crear directorios necesarios
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/jobs", exist_ok=True)
    os.makedirs("data/cvs", exist_ok=True)
    os.makedirs("data/results", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    main()
