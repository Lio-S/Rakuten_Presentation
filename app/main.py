#!/usr/bin/env python3
"""
Main.py simplifié pour Streamlit - Challenge Rakuten
Initialise seulement le pipeline et vérifie les prérequis
"""

import os
import sys
from pathlib import Path

# Ajout du chemin pour les imports
sys.path.append(str(Path(__file__).parent))

from preprocess import ProductClassificationPipeline, PipelineConfig

def initialize_pipeline():
    """
    Initialise le pipeline pour Streamlit
    """
    print("🚀 Initialisation du pipeline Rakuten...")
    
    # Vérification des fichiers requis
    required_files = [
        'config.yaml'
        'data/X_train_update.csv',
        'data/Y_train_CVw08PX.csv',
        'data/X_test_update.csv'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f"❌ Fichiers manquants : {missing_files}")
        return None
    
    # Configuration du pipeline
    config = PipelineConfig.from_yaml('config.yaml')
    pipeline = ProductClassificationPipeline(config)
    
    # Préparation des données (chargement uniquement)
    print("📁 Chargement des données prétraitées...")
    try:
        pipeline.prepare_data(
            force_preprocess_image=False,
            force_preprocess_text=False
        )
        print("✅ Données chargées avec succès")
    except Exception as e:
        print(f"❌ Erreur chargement données : {str(e)}")
        return None
    
    # Vérification des modèles
    models_status = {}
    
    # Modèles image
    for model_name in ['xgboost', 'neural_net']:
        model_path = os.path.join('data/models', model_name)
        if os.path.exists(model_path):
            models_status[model_name] = "✅ Disponible"
        else:
            models_status[model_name] = "❌ Manquant"
    
    # Modèle texte
    svm_path = 'data/models/SVM/model.pkl'
    if os.path.exists(svm_path):
        models_status['SVM'] = "✅ Disponible"
    else:
        models_status['SVM'] = "❌ Manquant"
    
    # Affichage du statut
    print("\n📊 Statut des modèles :")
    for model, status in models_status.items():
        print(f"  {model}: {status}")
    
    # Vérification des résultats
    results_files = [
        'data/results/image_models_comparison_results.csv',
        'data/results/text_model_results.csv',
        'data/reports/multimodal_comparison_results.csv'
    ]
    
    existing_results = [f for f in results_files if os.path.exists(f)]
    print(f"\n📈 Fichiers de résultats trouvés : {len(existing_results)}/{len(results_files)}")
    
    # Création des répertoires nécessaires
    directories = [
        'data/reports',
        'data/explanations',
        'data/rapports',
        'data/erreurs',
        'data/results',
        'temp_uploads'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✅ Pipeline initialisé et prêt pour Streamlit")
    return pipeline

def check_streamlit_requirements():
    """
    Vérifie les dépendances pour Streamlit
    """
    required_packages = [
        'streamlit',
        'plotly',
        'seaborn',
        'PIL'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Packages manquants pour Streamlit : {missing_packages}")
        print("💡 Installez avec : pip install streamlit plotly seaborn pillow")
        return False
    
    return True

def main():
    """
    Point d'entrée principal
    """
    print("🛍️ Challenge Rakuten - Initialisation pour Streamlit")
    print("=" * 50)
    
    # Vérification des prérequis
    if not check_streamlit_requirements():
        return
    
    # Initialisation du pipeline
    pipeline = initialize_pipeline()
    
    if pipeline is None:
        print("❌ Échec de l'initialisation")
        return
    
    print("\n🎯 Pipeline prêt !")
    print("💡 Pour lancer l'application : streamlit run app.py")
    print("🌐 L'application sera disponible sur : http://localhost:8501")

if __name__ == "__main__":
    main()