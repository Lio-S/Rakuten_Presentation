"""
Utilitaire pour gérer les chemins de manière cohérente dans tout le projet
"""
import os
from pathlib import Path

def get_project_root():
    """
    Retourne le répertoire racine du projet (Rakuten_Presentation)
    en remontant depuis le fichier actuel
    """
    # Obtenir le chemin du fichier actuel (utils_path.py)
    current_file = Path(__file__).resolve()
    
    # Le fichier est dans app/, donc on remonte d'un niveau pour obtenir la racine
    project_root = current_file.parent.parent
    
    return project_root

def get_data_path():
    """Retourne le chemin vers le dossier data"""
    return get_project_root() / "data"

def get_app_path():
    """Retourne le chemin vers le dossier app"""
    return get_project_root() / "app"

# Chemins principaux
PROJECT_ROOT = get_project_root()
DATA_DIR = get_data_path()
APP_DIR = get_app_path()

# Sous-dossiers data
IMAGES_DIR = DATA_DIR / "images"
TRAIN_IMAGES_DIR = IMAGES_DIR / "image_train"
TEST_IMAGES_DIR = IMAGES_DIR / "image_test"

MODELS_DIR = DATA_DIR / "models"
RESULTS_DIR = DATA_DIR / "results"
REPORTS_DIR = DATA_DIR / "reports"
RAPPORTS_DIR = DATA_DIR / "rapports"
ERREURS_DIR = DATA_DIR / "erreurs"
PROCESSED_DATA_DIR = DATA_DIR / "processed_data"
PREDICTIONS_DIR = DATA_DIR / "predictions"
LOGS_DIR = DATA_DIR / "logs"
INDICES_DIR = DATA_DIR / "indices"
EXPLANATIONS_DIR = DATA_DIR / "explanations"

# Fichiers CSV principaux
X_TRAIN_FILE = DATA_DIR / "X_train_update.csv"
Y_TRAIN_FILE = DATA_DIR / "Y_train_CVw08PX.csv"
X_TEST_FILE = DATA_DIR / "X_test_update.csv"

# Fichiers de configuration
CONFIG_FILE = APP_DIR / "config.yaml"

def ensure_directories():
    """Crée tous les répertoires nécessaires s'ils n'existent pas"""
    directories = [
        DATA_DIR,
        IMAGES_DIR,
        TRAIN_IMAGES_DIR,
        TEST_IMAGES_DIR,
        MODELS_DIR,
        RESULTS_DIR,
        REPORTS_DIR,
        RAPPORTS_DIR,
        ERREURS_DIR,
        PROCESSED_DATA_DIR,
        PREDICTIONS_DIR,
        LOGS_DIR,
        INDICES_DIR,
        EXPLANATIONS_DIR,
        APP_DIR / "temp_uploads"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

def get_model_path(model_type):
    """Retourne le chemin vers un modèle spécifique"""
    return MODELS_DIR / model_type

def get_result_file(filename):
    """Retourne le chemin complet vers un fichier de résultats"""
    return RESULTS_DIR / filename

def get_report_file(filename):
    """Retourne le chemin complet vers un fichier de rapport"""
    return REPORTS_DIR / filename

def get_rapport_file(filename):
    """Retourne le chemin complet vers un fichier de rapport (dossier rapports)"""
    return RAPPORTS_DIR / filename

def get_erreur_file(filename):
    """Retourne le chemin complet vers un fichier d'erreurs"""
    return ERREURS_DIR / filename

def get_processed_data_file(filename):
    """Retourne le chemin complet vers un fichier de données prétraitées"""
    return PROCESSED_DATA_DIR / filename

def get_prediction_file(filename):
    """Retourne le chemin complet vers un fichier de prédictions"""
    return PREDICTIONS_DIR / filename

def debug_paths():
    """Affiche tous les chemins pour debug"""
    print("=== Configuration des chemins ===")
    print(f"Répertoire de travail actuel: {Path.cwd()}")
    print(f"Racine du projet: {PROJECT_ROOT}")
    print(f"Dossier data: {DATA_DIR}")
    print(f"Dossier app: {APP_DIR}")
    print(f"Config file: {CONFIG_FILE}")
    print(f"X_train: {X_TRAIN_FILE}")
    print(f"Existe? {X_TRAIN_FILE.exists()}")
    print("================================")

# Créer les répertoires au chargement du module
ensure_directories()