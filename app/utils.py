import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def setup_logger(name, log_level=logging.INFO):
    """
    Configure un logger unique pour éviter les doublons
    
    Args:
        name: Nom du logger
        log_level: Niveau de logging
    
    Returns:
        Logger configuré
    """
    # Récupérer le logger par nom
    logger = logging.getLogger(name)
    
    # Si le logger existe déjà et est configuré, le retourner tel quel
    if logger.hasHandlers():
        return logger
        
    # Supprimer tous les handlers existants
    logger.handlers.clear()
    
    # Configurer le nouveau handler
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    # Ajouter le handler et définir le niveau
    logger.addHandler(handler)
    logger.setLevel(log_level)
    
    # Éviter la propagation des logs aux loggers parents
    logger.propagate = False
    
    return logger


def generate_prediction_report(pipeline, predictions, probabilities):
    """
    Génère un rapport de prédictions avec informations originales.
    """
    if hasattr(pipeline.preprocessed_data, 'test_split_indices'):
        test_indices = pipeline.preprocessed_data['test_split_indices']
    else:
        # Créer des indices par défaut si non disponibles
        test_indices = np.arange(len(predictions))
        
    test_mapping = {idx: f'img_{idx}' for idx in test_indices}
    
    print("len(test_indices) = ", len(test_indices))
    print("len(imageid)     = ", len([test_mapping.get(idx, 'N/A') for idx in test_indices]))
    print("len(predictions) = ", len(predictions))
    print("probabilities.shape =", probabilities.shape)
    print("len(y_test_split) = ", len(pipeline.preprocessed_data.get('y_test_split', [])))

    report = pd.DataFrame({
        'original_index': test_indices,
        'imageid': [test_mapping.get(idx, 'N/A') for idx in test_indices],
        'predicted_category': predictions,
        'prediction_probability': np.max(probabilities, axis=1),
        'true_category': pipeline.preprocessed_data.get('y_test_split', None)
    })
    
    report['predicted_category_name'] = [
        pipeline.category_names.get(cat, f'Catégorie {cat}') 
        for cat in report['predicted_category']
    ]
    
    return report

def analyze_prediction_errors(pipeline, predictions, true_labels):
    """
    Analyse détaillée des erreurs de prédiction.
    """
    error_mask = predictions != true_labels
    
    if hasattr(pipeline.preprocessed_data, 'test_indices') and hasattr(pipeline.preprocessed_data, 'test_mapping'):
        test_indices = pipeline.preprocessed_data['test_indices'][error_mask]
        test_mapping = pipeline.preprocessed_data.get('test_mapping', {})
    else:
        test_indices = np.arange(len(predictions))[error_mask]
        test_mapping = {idx: f'img_{idx}' for idx in test_indices}
    
    error_analysis = pd.DataFrame({
        'original_index': test_indices,
        'imageid': [test_mapping.get(idx, 'N/A') for idx in test_indices],
        'predicted_category': predictions[error_mask],
        'true_category': true_labels[error_mask]
    })
    
    error_analysis['predicted_category_name'] = [
        pipeline.category_names.get(cat, f'Catégorie {cat}') 
        for cat in error_analysis['predicted_category']
    ]
    error_analysis['true_category_name'] = [
        pipeline.category_names.get(cat, f'Catégorie {cat}') 
        for cat in error_analysis['true_category']
    ]
    
    return error_analysis    
    
def plot_prediction_distribution(report,model_name):
    """Visualisation de la distribution des prédictions"""
    plt.figure(figsize=(12, 6))
    report['predicted_category_name'].value_counts().plot(kind='bar')
    plt.title(f'Distribution des Catégories Prédites - {model_name}')
    plt.xlabel('Catégorie')
    plt.ylabel('Nombre de Prédictions')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()    