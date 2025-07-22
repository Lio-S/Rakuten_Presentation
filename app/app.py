import streamlit as st
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
from PIL import Image
import os
import sys
import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# from pathlib import Path

from utils_path import *
import main #(chargement des données)
from utils import safe_read_csv

# Constantes de chemins additionnelles
TRAIN_IMAGES_DIR = DATA_DIR / "images" / "image_train"
TEST_IMAGES_DIR = DATA_DIR / "images" / "image_test"
X_TRAIN_FILE = DATA_DIR / "X_train_update.csv"
Y_TRAIN_FILE = DATA_DIR / "Y_train_CVw08PX.csv"
TEST_FILE = DATA_DIR / "X_test_update.csv"

# Configuration de la page
st.set_page_config(
    page_title="Challenge Rakuten - Classification Multimodale",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache pour le pipeline
@st.cache_resource
def load_pipeline():
    """Charge le pipeline et les modèles (mis en cache)"""
    
    try:
    #     # Vérifier que les fichiers existent
    #     st.write(f"**config.yaml existe:** {'✅' if CONFIG_FILE.exists() else '❌'}")
    #     st.write(f"**preprocess.py existe:** {'✅' if (APP_DIR / 'preprocess.py').exists() else '❌'}")
                
        # Ajouter le répertoire du script au PATH Python
        if str(APP_DIR) not in sys.path:
            sys.path.insert(0, str(APP_DIR))
        
        from preprocess import ProductClassificationPipeline, PipelineConfig
        
        if not CONFIG_FILE.exists():
            st.error(f"❌ config.yaml non trouvé dans {APP_DIR}")
            return None
            
        config = PipelineConfig.from_yaml(str(CONFIG_FILE))
        pipeline = ProductClassificationPipeline(config)
        
        # Charger les données pré-traitées
        pipeline.prepare_data(force_preprocess_image=False, force_preprocess_text=False)
        
        # Charger les modèles pré-entraînés
        models_loaded = []
        
        try:
            pipeline.load_model('xgboost')
            models_loaded.append("XGBoost")
        except Exception as e:
            st.warning(f"⚠️ XGBoost non disponible: {e}")
        
        try:
            pipeline.load_model('neural_net')
            models_loaded.append("Neural Network")
        except Exception as e:
            st.warning(f"⚠️ Neural Net non disponible: {e}")
        
        try:
            pipeline.load_text_model('SVM')
            models_loaded.append("SVM Texte")
        except Exception as e:
            st.warning(f"⚠️ SVM Texte non disponible: {e}")
        
        if not models_loaded:
            st.error("❌ Aucun modèle n'a pu être chargé. Vérifiez que main.py a été exécuté.")
            return None
        
        return pipeline
        
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du pipeline: {e}")
        st.info("💡 Assurez-vous d'avoir exécuté main.py pour générer les modèles et données.")
        return None

# Cache pour les données de résultats
@st.cache_data
def load_results_data():
    """Charge les données de résultats et rapports"""
    results = {}
    
    # Résultats de comparaison des modèles
    results_files = {
        'image_models': get_result_file('models_comparaison_results.csv'),
        'text_model': get_result_file('text_model_results.csv'),
        'multimodal': get_report_file('multimodal_comparison_results.csv')
    }
    
    for key, file_path in results_files.items():
        if file_path.exists:
            try:
                results[key] = safe_read_csv(str(file_path))
            except Exception as e:
                st.warning(f"Erreur lecture {file_path}: {e}")
    
    # Rapports détaillés
    rapport_files = {
        'rapport_xgboost': get_rapport_file('rapport_xgboost.csv'),
        'rapport_neural_net': get_rapport_file('rapport_neural_net.csv'),
        'rapport_text_SVM': get_rapport_file('rapport_text_SVM.csv'),
        'erreurs_xgboost': get_erreur_file('erreurs_xgboost.csv'),
        'erreurs_neural_net': get_erreur_file('erreurs_neural_net.csv'),
        'erreurs_text_SVM': get_erreur_file('erreurs_text_SVM.csv')
    }
    
    for key, file_path in rapport_files.items():
        if file_path.exists():
            try:
                results[key] = safe_read_csv(str(file_path))
            except Exception as e:
                st.warning(f"Erreur lecture {file_path}: {e}")
    
    return results

def check_columns_and_get_mapping(df, expected_columns, file_type="rapport"):
    """
    Vérifie les colonnes disponibles et retourne un mapping avec création de colonnes manquantes
    
    Args:
        df: DataFrame à analyser
        expected_columns: Liste des colonnes attendues
        file_type: Type de fichier ("rapport" ou "erreurs")
    """
    available_columns = df.columns.tolist()
    column_mapping = {}
    
    # Mapping basé sur les vraies colonnes générées par main.py
    if file_type == "rapport":
        possible_mappings = {
            'true_class_name': ['true_class_name', 'true_category_name', 'true_label'],
            'predicted_class_name': ['predicted_class_name', 'predicted_category_name', 'predicted_label'],
            'correct': ['correct', 'is_correct'],
            'confidence': ['prediction_probability', 'confidence', 'prob_max', 'max_proba'],  # prediction_probability en premier
            'true_category': ['true_category', 'true_label'],
            'predicted_category': ['predicted_category', 'predicted_label']
        }
    else:  # file_type == "erreurs"
        possible_mappings = {
            'true_class_name': ['true_class_name', 'true_category_name', 'true_label'],
            'predicted_class_name': ['predicted_class_name', 'predicted_category_name', 'predicted_label'],
            'correct': ['correct', 'is_correct'],
            'confidence': ['prediction_probability', 'confidence', 'prob_max', 'max_proba'],  # prediction_probability en premier
            'true_category': ['true_category', 'true_label'],
            'predicted_category': ['predicted_category', 'predicted_label']
        }
    
    # Créer une copie du DataFrame pour ajouter les colonnes manquantes
    df_processed = df.copy()
    
    # Mapping des codes vers les noms
    category_names = {
            10: "Livres occasion",
            40: "Jeux consoles neuf", 
            50: "Accessoires gaming",
            60: "Consoles de jeux",
            1140: "Objets pop culture",
            1160: "Cartes de jeux",
            1180: "Jeux de rôle et figurines",
            1280: "Jouets enfant",
            1300: "Modélisme",
            1281: "Jeux enfant", 
            1301: "Lingerie enfant et jeu de bar",
            1302: "Jeux et accessoires de plein air",
            1320: "Puériculture",
            1560: "Mobilier",
            1920: "Linge de maison",
            1940: "Épicerie",
            2060: "Décoration",
            2220: "Animalerie",
            2280: "Journaux et revues occasion",
            2403: "Lots livres et magazines",
            2462: "Console et Jeux vidéos occasion",
            2522: "Fournitures papeterie",
            2582: "Mobilier et accessoires de jardin",
            2583: "Piscine et accessoires",
            2585: "Outillage de jardin",
            2705: "Livres neufs",
            2905: "Jeux PC en téléchargement"
    }
    
    # Mapping détaillé avec descriptions et emoji
    def get_category_description(code):
        descriptions = {
            10: {"name": "Livres occasion", "emoji": "📚", "desc": "Romans, BD, essais d'occasion"},
            40: {"name": "Jeux consoles neuf", "emoji": "🆕", "desc": "Jeux neufs pour consoles"},
            50: {"name": "Accessoires gaming", "emoji": "🎧", "desc": "Casques, manettes, équipements gamer"},
            60: {"name": "Consoles de jeux", "emoji": "🎮", "desc": "PlayStation, Xbox, Nintendo et autres"},
            1300: {"name": "Modélisme", "emoji": "✈️", "desc": "Maquettes, trains miniatures, dioramas"},
            1140: {"name": "Objets pop culture", "emoji": "🧙‍♂️", "desc": "Figurines, goodies, objets collectors"},
            1160: {"name": "Cartes de jeux", "emoji": "🃏", "desc": "Cartes Pokémon, Magic, Yu-Gi-Oh!"},
            1180: {"name": "Jeux de rôle et figurines", "emoji": "🐉", "desc": "Warhammer, Donjons & Dragons, figurines"},
            1280: {"name": "Jouets enfant", "emoji": "🧸", "desc": "Jouets pour tous âges, éducatifs ou ludiques"},
            1320: {"name": "Puériculture", "emoji": "👶", "desc": "Biberons, poussettes, produits bébé"},
            1560: {"name": "Mobilier", "emoji": "🪑", "desc": "Meubles et accessoires pour toutes les pièces de la maison"},
            1920: {"name": "Linge de maison", "emoji": "🛏️", "desc": "Draps, serviettes, couvertures"},
            1940: {"name": "Épicerie", "emoji": "🛒", "desc": "Produits alimentaires et boissons"},
            2060: {"name": "Décoration", "emoji": "🖼️", "desc": "Objets déco, cadres, bougies"},
            2220: {"name": "Animalerie", "emoji": "🐾", "desc": "Accessoires pour chiens, chats et NAC"},
            2280: {"name": "Journaux et revues occasion", "emoji": "📰", "desc": "Magazines, journaux, revues d'occasion"},
            1281: {"name": "Jeux enfant", "emoji": "🧩", "desc": "Jeux d'éveil, de construction ou de société"},
            1301: {"name": "Lingerie enfant et jeu de bar", "emoji": "🧦", "desc": "Chaussettes ludiques pour enfants, billard babyfoot et flechettes"},
            1302: {"name": "Jeux et accessoires de plein air", "emoji": "🏸", "desc": "Trottinettes, ballons, jeux d'extérieur"},
            2403: {"name": "Lots livres et magazines", "emoji": "📦", "desc": "Packs de livres, collections de magazines"},
            2462: {"name": "Console et Jeux vidéos occasion", "emoji": "💿", "desc": "Jeux vidéo d'occasion pour toutes consoles"},
            2522: {"name": "Fournitures papeterie", "emoji": "🖊️", "desc": "Stylos, cahiers, articles scolaires"},
            2582: {"name": "Mobilier et accessoire de jardin", "emoji": "🌳", "desc": "Tables, chaises, bancs d'extérieur"},
            2583: {"name": "Piscine et accessoires", "emoji": "🏊", "desc": "Piscines gonflables, jeux d'eau"},
            2585: {"name": "Outillage de jardin", "emoji": "🛠️", "desc": "Outils, tondeuses, équipements jardin"},
            2705: {"name": "Livres neufs", "emoji": "📖", "desc": "Romans, essais, albums neufs"},
            2905: {"name": "Jeux PC en téléchargement", "emoji": "🖥️", "desc": "Jeux pour ordinateur, clefs numériques"}
        }

        
        return descriptions.get(code, {
            "name": f"Code_{code}", 
            "emoji": "❓", 
            "desc": "Catégorie non documentée"
        })
    
    # Fonction pour améliorer la sélection d'exemples
    def get_diverse_examples():
        """Sélectionne des exemples en s'assurant d'avoir une bonne diversité de catégories"""
        try:
            # Charger les données
            X_train_df = safe_read_csv('../data/X_train_update.csv')
            Y_train_df = safe_read_csv('../data/Y_train_CVw08PX.csv')
            
            # Récupérer les indices du test_split
            if hasattr(pipeline, 'preprocessed_data') and 'test_split_indices' in pipeline.preprocessed_data:
                test_split_indices = pipeline.preprocessed_data['test_split_indices']
            else:
                n_total = len(X_train_df)
                test_split_indices = X_train_df.index[-int(0.2 * n_total):]
            
            # Grouper par catégorie
            available_indices = [idx for idx in test_split_indices if idx in X_train_df.index and idx in Y_train_df.index]
            
            category_groups = {}
            for idx in available_indices:
                if idx in Y_train_df.index:
                    category = Y_train_df.loc[idx, 'prdtypecode']
                    if category not in category_groups:
                        category_groups[category] = []
                    category_groups[category].append(idx)
            
            # Sélectionner 2-3 exemples par catégorie disponible
            selected_examples = []
            
            for category, indices in category_groups.items():
                # Vérifier que les images existent
                valid_indices = []
                for idx in indices[:10]:  # Vérifier les 10 premiers
                    row = X_train_df.loc[idx]
                    image_file = f"image_{row['imageid']}_product_{row['productid']}.jpg"
                    image_path = os.path.join('../data/images/image_train', image_file)
                    if image_path.exists():
                        valid_indices.append(idx)
                
                # Prendre 2-3 exemples valides par catégorie
                if len(valid_indices) > 0:
                    n_samples = min(3, len(valid_indices))
                    selected_indices = np.random.choice(valid_indices, size=n_samples, replace=False)
                    selected_examples.extend(selected_indices)
            
            # Créer les exemples
            samples = []
            for idx in selected_examples:
                row = X_train_df.loc[idx]
                label = Y_train_df.loc[idx]
                
                designation = str(row.get('designation', '')).strip()
                description = str(row.get('description', '')).strip()
                text = f"{designation} {description}".strip()
                
                image_file = f"image_{row['imageid']}_product_{row['productid']}.jpg"
                image_path = os.path.join('../data/images/image_train', image_file)
                
                class_code = label['prdtypecode']
                category_info = get_category_description(class_code)
                
                if len(text) > 20:
                    samples.append({
                        'text': text,
                        'image_path': image_path,
                        'class_name': category_info['name'],
                        'class_code': class_code,
                        'class_emoji': category_info['emoji'],
                        'class_desc': category_info['desc'],
                        'imageid': row['imageid'],
                        'productid': row['productid'],
                        'index': idx,
                        'designation': designation,
                        'description': description
                    })
            
            return samples, category_groups
            
        except Exception as e:
            st.error(f"❌ Erreur sélection exemples diversifiés: {e}")
            return [], {}
    
    for expected_col in expected_columns:
        # Chercher la colonne existante
        found = False
        for possible_name in possible_mappings.get(expected_col, [expected_col]):
            if possible_name in available_columns:
                column_mapping[expected_col] = possible_name
                found = True
                break
        
        # Si pas trouvée, essayer de créer la colonne
        if not found:
            if expected_col == 'correct' and 'true_category' in available_columns and 'predicted_category' in available_columns:
                # Créer la colonne 'correct' en comparant true_category et predicted_category
                df_processed['correct'] = df_processed['true_category'] == df_processed['predicted_category']
                column_mapping['correct'] = 'correct'
                st.info("✅ Colonne 'correct' créée automatiquement")
                found = True
                
            elif expected_col == 'confidence':
                # Vérifier d'abord si prediction_probability existe
                if 'prediction_probability' in available_columns:
                    column_mapping['confidence'] = 'prediction_probability'
                    found = True
                elif file_type == "erreurs":
                    # Pour les fichiers d'erreurs, la confidence peut ne pas exister
                    st.info("ℹ️ Colonne 'confidence' non disponible dans le fichier d'erreurs")
                    # Ne pas créer de colonne par défaut, juste ignorer
                    found = False
                else:
                    # Pour les rapports, essayer autres noms
                    found = False
                
            elif expected_col == 'true_class_name':
                # Essayer de créer à partir de true_category_name ou true_category
                if 'true_category_name' in available_columns:
                    column_mapping['true_class_name'] = 'true_category_name'
                    found = True
                elif 'true_category' in available_columns:
                    df_processed['true_class_name'] = df_processed['true_category'].map(category_names)
                    column_mapping['true_class_name'] = 'true_class_name'
                    st.info("✅ Colonne 'true_class_name' créée à partir de 'true_category'")
                    found = True
                    
            elif expected_col == 'predicted_class_name':
                # Essayer de créer à partir de predicted_category_name ou predicted_category
                if 'predicted_category_name' in available_columns:
                    column_mapping['predicted_class_name'] = 'predicted_category_name'
                    found = True
                elif 'predicted_category' in available_columns:
                    df_processed['predicted_class_name'] = df_processed['predicted_category'].map(category_names)
                    column_mapping['predicted_class_name'] = 'predicted_class_name'
                    st.info("✅ Colonne 'predicted_class_name' créée à partir de 'predicted_category'")
                    found = True
        
        # Si toujours pas trouvée, warning seulement si c'est critique
        if not found and expected_col != 'confidence':
            st.warning(f"⚠️ Colonne '{expected_col}' non trouvée. Colonnes disponibles: {available_columns}")
    
    return column_mapping, df_processed

# Sidebar pour navigation
st.sidebar.title("🛍️ Navigation")
page = st.sidebar.selectbox(
    "Choisir une page",
    ["🏠 Accueil", "📊 Résultats Globaux", "🖼️ Analyse Images", "📝 Analyse Texte", 
     "🔗 Analyse Multimodale", "🧪 Test Nouvelles Données", "🎯 Explicabilité"]
)

# Charger le pipeline
pipeline = load_pipeline()
results_data = load_results_data()

# Vérification que le pipeline est disponible
if pipeline is None:
    st.error("❌ Pipeline non disponible")
    st.info("💡 Veuillez exécuter main.py pour générer les modèles et données avant d'utiliser cette application.")
    st.stop()  # Arrête l'exécution de l'app

# ==================== PAGE ACCUEIL ====================
if page == "🏠 Accueil":
    st.title("🛍️ Challenge Rakuten - Classification Multimodale")
    st.markdown("---")
    
    # Présentation du projet
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## 🎯 Objectif du Challenge
        
        Ce projet vise à **classifier automatiquement les produits Rakuten** en utilisant :
        - 📝 **Texte** (designation + description)
        - 🖼️ **Images** des produits
        - 🔗 **Fusion multimodale** des deux approches
        
        ### 📊 Données du Challenge
        """)
        
        # Statistiques générales
        stats_col1, stats_col2, stats_col3 = st.columns(3)
        with stats_col1:
            st.metric("📦 Produits Total", "84,916")
            st.metric("🏋️ Train Balancé", "54,000")
        with stats_col2:
            st.metric("🎯 Classes", "27")
            st.metric("🧪 Test Split", "16,984")
        with stats_col3:
            st.metric("🖼️ Images", "84,916")
            st.metric("📝 Textes", "84,916")
    
    with col2:
        st.markdown("""
        ### 🏆 Meilleurs Résultats
        """)
        
        # Afficher les meilleurs résultats si disponibles
        if 'multimodal' in results_data and not results_data['multimodal'].empty:
            try:
                best_model = results_data['multimodal'].loc[results_data['multimodal']['weighted_f1'].idxmax()]
                st.success(f"🥇 **{best_model.name}**")
                st.metric("F1-Score", f"{best_model['weighted_f1']:.3f}")
                st.metric("Accuracy", f"{best_model['accuracy']:.3f}")
            except Exception as e:
                st.info("Résultats multimodaux en cours de traitement...")
        else:
            st.info("Résultats multimodaux en cours de traitement...")
    
    # Distribution des classes
    st.markdown("### 📊 Distribution des Classes")
    
    # Données de distribution (basées sur vos outputs)
    class_data = {
        'Classe': [1, 4, 5, 6, 13, 114, 116, 118, 128, 132, 156, 192, 194, 206, 222, 228, 1281, 1301, 1302, 2403, 2462, 2522, 2582, 2583, 2585, 2705, 2905],
        'Nom': ["Livres occasion","Jeux consoles neuf", "Accessoires gaming","Consoles de jeux","Modélisme","Objets pop culture","Cartes de jeux","Jeux de rôle et figurines","Jouets enfant","Puériculture","Mobilier","Linge de maison","Épicerie","Décoration","Animalerie","Journaux et revues occasion","Jeux enfant", "Lingerie enfant et jeu de bar","Jeux et accessoires de plein air","Lots livres et magazines","Console et Jeux vidéos occasion","Fournitures papeterie","Mobilier et accessoires de jardin","Piscine et accessoires","Outillage de jardin","Livres neufs","Jeux PC en téléchargement"
],
        'Échantillons': [3116, 2508, 1681, 832, 2671, 3953, 764, 4870, 2070, 5045, 807, 2491, 3241, 5073, 4303, 803, 4993, 824, 4760, 4774, 1421, 4989, 2589, 10209, 2496, 2761, 872],
        'Pourcentage': [3.67, 2.95, 1.98, 0.98, 3.15, 4.66, 0.90, 5.74, 2.44, 5.94, 0.95, 2.93, 3.82, 5.97, 5.07, 0.95, 5.88, 0.97, 5.61, 5.62, 1.67, 5.88, 3.05, 12.02, 2.94, 3.25, 1.03]
    }
    
    df_classes = pd.DataFrame(class_data)
    
    if st.button("🔍 Debug des chemins"):
        debug_paths()
        
    # Graphique interactif
    fig = px.bar(df_classes, x='Nom', y='Échantillons', 
                 title="Distribution des Classes (Données Originales)",
                 labels={'Nom': 'Catégorie', 'Échantillons': 'Nombre d\'échantillons'})
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

# ==================== PAGE RÉSULTATS GLOBAUX ====================
elif page == "📊 Résultats Globaux":
    st.title("📊 Résultats Globaux de Classification")
    st.markdown("---")
    
    # Comparaison des modèles
    if 'image_models' in results_data and 'text_model' in results_data:
        st.subheader("🏆 Comparaison des Performances")
        
        # Préparer les données pour la comparaison
        comparaison_data = []
        
        # Modèles image
        for model_name, row in results_data['image_models'].iterrows():
            comparaison_data.append({
                'Modèle': model_name,
                'Type': 'Image',
                'Accuracy': row['accuracy'],
                'F1-Score': row['weighted_f1'],
                'Précision': row.get('weighted_precision', 0),
                'Rappel': row.get('weighted_recall', 0)
            })
        
        # Modèle texte
        for model_name, row in results_data['text_model'].iterrows():
            comparaison_data.append({
                'Modèle': model_name,
                'Type': 'Texte',
                'Accuracy': row['accuracy'],
                'F1-Score': row['weighted_f1'],
                'Précision': row.get('weighted_precision', 0),
                'Rappel': row.get('weighted_recall', 0)
            })
        
        # Modèles multimodaux
        if 'multimodal' in results_data and not results_data['multimodal'].empty:
            for model_name, row in results_data['multimodal'].iterrows():
                model_type = 'Multimodal' if 'multimodal' in str(row.get('model_type', '')) else 'Fusion'
                comparaison_data.append({
                    'Modèle': model_name,
                    'Type': model_type,
                    'Accuracy': row['accuracy'],
                    'F1-Score': row['weighted_f1'],
                    'Précision': row.get('weighted_precision', 0),
                    'Rappel': row.get('weighted_recall', 0)
                })
        
        df_comparaison = pd.DataFrame(comparaison_data)
        
        if not df_comparaison.empty:
            # Graphique de comparaison
            fig = px.scatter(df_comparaison, x='Accuracy', y='F1-Score', 
                            color='Type', size='Précision',
                            hover_data=['Modèle', 'Rappel'],
                            title="Performance des Modèles (Accuracy vs F1-Score)")
            st.plotly_chart(fig, use_container_width=True)
            
            # Tableau récapitulatif
            st.subheader("📋 Tableau Récapitulatif")
            df_display = df_comparaison.round(3)
            st.dataframe(df_display, use_container_width=True)
    
    # Métriques par classe (si disponible)
    if 'rapport_xgboost' in results_data:
        st.subheader("📊 Performance par Classe")
        
        rapport = results_data['rapport_xgboost']
        
        # Vérifier les colonnes disponibles
        column_mapping, rapport_processed = check_columns_and_get_mapping(rapport, ['true_class_name', 'correct', 'confidence'], "rapport")
        
        if 'true_class_name' in column_mapping and 'correct' in column_mapping:
            # Calculer les métriques par classe
            try:
                # Utiliser le DataFrame traité
                true_col = column_mapping['true_class_name']
                correct_col = column_mapping['correct']
                conf_col = column_mapping.get('confidence', correct_col)
                
                class_metrics = rapport_processed.groupby(true_col).agg({
                    correct_col: 'mean',
                    conf_col: 'mean'
                }).round(3)
                
                class_metrics.columns = ['Accuracy', 'Confiance Moyenne']
                
                # Trier par accuracy
                class_metrics = class_metrics.sort_values('Accuracy', ascending=False)
                
                st.dataframe(class_metrics, use_container_width=True)
            except Exception as e:
                st.error(f"Erreur calcul métriques par classe: {e}")
                st.info("Colonnes disponibles: " + str(rapport.columns.tolist()))

# ==================== PAGE ANALYSE IMAGES ====================
elif page == "🖼️ Analyse Images":
    st.title("🖼️ Analyse des Modèles Image")
    st.markdown("---")
    
    # Sélection du modèle
    model_choice = st.selectbox("Choisir le modèle image", ["xgboost", "neural_net"])
    
    if f'rapport_{model_choice}' in results_data:
        rapport = results_data[f'rapport_{model_choice}']
        
        # Vérifier les colonnes disponibles
        column_mapping, rapport_processed = check_columns_and_get_mapping(rapport, ['correct', 'confidence', 'true_class_name', 'predicted_class_name'], "rapport")
        
        if 'correct' in column_mapping and 'confidence' in column_mapping:
            # Métriques globales
            st.subheader("📊 Métriques Globales")
            col1, col2, col3, col4 = st.columns(4)
            
            correct_col = column_mapping['correct']
            conf_col = column_mapping['confidence']
            
            with col1:
                accuracy = rapport_processed[correct_col].mean()
                st.metric("Accuracy", f"{accuracy:.3f}")
            
            with col2:
                confidence = rapport_processed[conf_col].mean()
                st.metric("Confiance Moyenne", f"{confidence:.3f}")
            
            with col3:
                high_conf = (rapport_processed[conf_col] > 0.8).sum()
                st.metric("Haute Confiance (>0.8)", f"{high_conf}")
            
            with col4:
                low_conf = (rapport_processed[conf_col] < 0.5).sum()
                st.metric("Faible Confiance (<0.5)", f"{low_conf}")
            
            # Distribution des confiances
            st.subheader("📈 Distribution des Confiances")
            fig = px.histogram(rapport_processed, x=conf_col, nbins=50,
                              title="Distribution des Scores de Confiance")
            st.plotly_chart(fig, use_container_width=True)
            
            # Matrice de confusion (top 10 classes)
            if 'true_class_name' in column_mapping and 'predicted_class_name' in column_mapping:
                st.subheader("🎯 Matrice de Confusion (Top 10 Classes)")
                
                true_col = column_mapping['true_class_name']
                pred_col = column_mapping['predicted_class_name']
                
                # Prendre les 10 classes les plus fréquentes
                top_classes = rapport_processed[true_col].value_counts().head(10).index
                rapport_top = rapport_processed[rapport_processed[true_col].isin(top_classes)]
                
                if not rapport_top.empty:
                    confusion_matrix = pd.crosstab(rapport_top[true_col], 
                                                 rapport_top[pred_col])
                    
                    fig = px.imshow(confusion_matrix, text_auto=True,
                                   title="Matrice de Confusion (Top 10 Classes)")
                    st.plotly_chart(fig, use_container_width=True)
        
        # Analyse des erreurs
        if f'erreurs_{model_choice}' in results_data:
            st.subheader("❌ Analyse des Erreurs")
            erreurs = results_data[f'erreurs_{model_choice}']
            
            # Vérifier les colonnes des erreurs
            error_column_mapping, erreurs_processed = check_columns_and_get_mapping(erreurs, ['true_class_name', 'predicted_class_name', 'confidence'], "erreurs")
            
            if 'true_class_name' in error_column_mapping:
                true_col = error_column_mapping['true_class_name']
                
                # Erreurs par classe
                erreurs_par_classe = erreurs_processed.groupby(true_col).size().sort_values(ascending=False)
                
                fig = px.bar(x=erreurs_par_classe.index[:15], y=erreurs_par_classe.values[:15],
                            title="Nombre d'Erreurs par Classe (Top 15)")
                st.plotly_chart(fig, use_container_width=True)
                
                # Exemples d'erreurs
                st.subheader("🔍 Exemples d'Erreurs")
                sample_errors = erreurs_processed.sample(min(10, len(erreurs_processed)))
                
                for _, error in sample_errors.iterrows():
                    true_name = error.get(error_column_mapping.get('true_class_name', 'N/A'), 'N/A')
                    pred_name = error.get(error_column_mapping.get('predicted_class_name', 'N/A'), 'N/A')
                    
                    with st.expander(f"Erreur: {true_name} → {pred_name}"):
                        # Afficher la confiance seulement si elle existe
                        if 'confidence' in error_column_mapping:
                            conf_val = error.get(error_column_mapping['confidence'], 'N/A')
                            st.write(f"**Probabilité de prédiction**: {conf_val}")
                        elif 'prediction_probability' in error.index:
                            conf_val = error.get('prediction_probability', 'N/A')
                            st.write(f"**Probabilité de prédiction**: {conf_val}")
                        else:
                            st.write("**Probabilité de prédiction**: Non disponible")
                        
                        if 'error_type' in error:
                            st.write(f"**Type d'erreur**: {error.get('error_type', 'N/A')}")
                        
                        # Afficher des infos additionnelles disponibles
                        if 'imageid' in error:
                            st.write(f"**ID Image**: {error.get('imageid', 'N/A')}")
                        if 'original_index' in error:
                            st.write(f"**Index Original**: {error.get('original_index', 'N/A')}")
    else:
        st.warning(f"Rapport pour {model_choice} non disponible. Exécutez d'abord main.py pour générer les rapports.")

# ==================== PAGE ANALYSE TEXTE ====================
elif page == "📝 Analyse Texte":
    st.title("📝 Analyse du Modèle Texte (SVM)")
    st.markdown("---")
    
    if 'rapport_text_SVM' in results_data:
        rapport = results_data['rapport_text_SVM']
        
        # Vérifier les colonnes disponibles
        column_mapping, rapport_processed = check_columns_and_get_mapping(rapport, ['correct', 'confidence', 'text_sample'], "rapport")
        
        if 'correct' in column_mapping and 'confidence' in column_mapping:
            correct_col = column_mapping['correct']
            conf_col = column_mapping['confidence']
            
            # Métriques globales
            st.subheader("📊 Métriques Globales")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                accuracy = rapport_processed[correct_col].mean()
                st.metric("Accuracy", f"{accuracy:.3f}")
            
            with col2:
                confidence = rapport_processed[conf_col].mean()
                st.metric("Confiance Moyenne", f"{confidence:.3f}")
            
            with col3:
                high_conf = (rapport_processed[conf_col] > 0.8).sum()
                st.metric("Haute Confiance", f"{high_conf}")
            
            with col4:
                low_conf = (rapport_processed[conf_col] < 0.5).sum()
                st.metric("Faible Confiance", f"{low_conf}")
            
            # Analyse des textes
            st.subheader("📝 Analyse des Textes")
            
            # Longueur des textes
            if 'text_sample' in column_mapping:
                text_col = column_mapping['text_sample']
                rapport_processed['text_length'] = rapport_processed[text_col].str.len()
                
                fig = px.histogram(rapport_processed, x='text_length', nbins=50,
                                  title="Distribution des Longueurs de Texte")
                st.plotly_chart(fig, use_container_width=True)
                
                # Corrélation longueur vs confiance
                fig = px.scatter(rapport_processed, x='text_length', y=conf_col,
                                color=correct_col, title="Longueur du Texte vs Confiance")
                st.plotly_chart(fig, use_container_width=True)
            
            # Mots les plus fréquents dans les erreurs
            if 'erreurs_text_SVM' in results_data:
                st.subheader("🔍 Analyse des Erreurs Texte")
                erreurs = results_data['erreurs_text_SVM']
                
                # Vérifier les colonnes des erreurs
                error_column_mapping, erreurs_processed = check_columns_and_get_mapping(erreurs, ['true_class_name', 'predicted_class_name', 'confidence', 'text_sample'], "erreurs")
                
                # Quelques exemples d'erreurs
                st.write("**Exemples d'erreurs de classification :**")
                sample_errors = erreurs_processed.sample(min(5, len(erreurs_processed)))
                
                for _, error in sample_errors.iterrows():
                    true_class = error.get(error_column_mapping.get('true_class_name', 'N/A'), 'N/A')
                    pred_class = error.get(error_column_mapping.get('predicted_class_name', 'N/A'), 'N/A')
                    
                    with st.expander(f"Erreur: {true_class} → {pred_class}"):
                        # Afficher la confiance seulement si elle existe
                        if 'confidence' in error_column_mapping:
                            conf_val = error.get(error_column_mapping['confidence'], 'N/A')
                            st.write(f"**Confiance**: {conf_val}")
                        else:
                            st.write("**Confiance**: Non disponible")
                        
                        if 'text_sample' in error_column_mapping:
                            text_sample = error.get(error_column_mapping['text_sample'], 'N/A')
                            st.write(f"**Texte**: {str(text_sample)[:200]}...")
    else:
        st.warning("Rapport texte SVM non disponible. Exécutez d'abord main.py pour générer les rapports.")

# ==================== PAGE ANALYSE MULTIMODALE ====================
elif page == "🔗 Analyse Multimodale":
    st.title("🔗 Analyse Multimodale")
    st.markdown("---")
    
    if 'multimodal' in results_data and not results_data['multimodal'].empty:
        multimodal_results = results_data['multimodal']
        
        # Comparaison des stratégies de fusion
        st.subheader("🔀 Comparaison des Stratégies de Fusion")
        
        # Filtrer les résultats multimodaux
        fusion_results = multimodal_results[multimodal_results['model_type'] == 'multimodal']
        
        if not fusion_results.empty:
            # Graphique de comparaison
            fig = px.bar(fusion_results, x=fusion_results.index, y='weighted_f1',
                        title="Performance des Stratégies de Fusion")
            st.plotly_chart(fig, use_container_width=True)
            
            # Tableau détaillé
            st.dataframe(fusion_results[['accuracy', 'weighted_f1']].round(3), use_container_width=True)
        
        # Analyse de l'amélioration multimodale
        st.subheader("📈 Gain de la Fusion Multimodale")
        
        # Comparer avec les modèles individuels
        individual_results = multimodal_results[multimodal_results['model_type'] != 'multimodal']
        
        if not individual_results.empty:
            comparaison_fig = px.bar(
                x=multimodal_results.index,
                y=multimodal_results['weighted_f1'],
                color=multimodal_results['model_type'],
                title="Comparaison Individuel vs Multimodal"
            )
            st.plotly_chart(comparaison_fig, use_container_width=True)
    else:
        st.warning("Résultats multimodaux non disponibles. Exécutez d'abord main.py pour générer les analyses multimodales.")

# ==================== PAGE TEST NOUVELLES DONNÉES ====================
elif page == "🧪 Test Nouvelles Données":
    st.title("🧪 Test sur Nouvelles Données")
    st.markdown("---")
    
    # Explication des modes
    with st.expander("📖 Explication des modes de test"):
        st.markdown("""
        **🎲 Exemple test_split**: Utilise des données **non vues pendant l'entraînement** (20% des données d'origine). 
        Les labels sont connus, permettant une évaluation correcte des performances.
        
        **🏆 Exemple challenge**: Utilise les **vraies données de test du challenge**. 
        Les labels ne sont pas connus, simule une utilisation réelle.
        
        **✍️ Saisie manuelle**: Permet de tester avec vos propres données.
        
        **📁 Upload fichier**: Traite des fichiers CSV avec plusieurs exemples.
        """)
    
    st.markdown("---")
    
    # Sélection des paramètres
    st.subheader("⚙️ Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        model_type = st.selectbox("Modèle Image", ["xgboost", "neural_net"])
    
    # Fonction pour extraire infos d'une image sélectionnée
    def extract_info_from_image(image_name):
        """Extrait les informations produit à partir du nom d'image"""
        try:
            # Format: image_{imageid}_product_{productid}.jpg
            parts = image_name.replace('.jpg', '').replace('.jpeg', '').replace('.png', '')
            if 'image_' in parts and '_product_' in parts:
                imageid = parts.split('image_')[1].split('_product_')[0]
                productid = parts.split('_product_')[1]
                
                # Charger les données
                X_test_df = safe_read_csv(str(TEST_FILE))  
                
                # Chercher la ligne correspondante
                matching_row = X_test_df[
                    (X_test_df['imageid'].astype(str) == str(imageid)) & 
                    (X_test_df['productid'].astype(str) == str(productid))
                ]
                
                if not matching_row.empty:
                    row = matching_row.iloc[0]
                    text = f"{row.get('designation', '')} {row.get('description', '')}".strip()
                    return {
                        'text': text,
                        'imageid': imageid,
                        'productid': productid,
                        'designation': row.get('designation', ''),
                        'description': row.get('description', '')
                    }
            return None
        except Exception as e:
            st.error(f"Erreur extraction info image: {e}")
            return None

    with col2:
        fusion_strategy = st.selectbox("Stratégie de Fusion", 
                                      ["mean", "product", "weighted", "confidence_weighted"])
    
    # Interface de test
    st.subheader("📝 Saisie des Données")
    
    # Modes de test
    test_mode = st.radio("Mode de test", 
                        ["🎲 Exemple test_split", "🏆 Exemple challenge", "✍️ Saisie manuelle"])
    
    if test_mode == "🎲 Exemple test_split":
        st.info("📊 **Données test_split** : Échantillons non vus pendant l'entraînement (labels connus)")
        
        # Bouton pour générer un nouvel exemple
        if st.button("🎲 Générer nouvel exemple test_split"):
            # Charger des exemples diversifiés comme dans la page explicabilité
            try:
                X_train_df = safe_read_csv(str(X_TRAIN_FILE))
                Y_train_df = safe_read_csv(str(Y_TRAIN_FILE))
                
                if hasattr(pipeline, 'preprocessed_data') and 'test_split_indices' in pipeline.preprocessed_data:
                    test_split_indices = pipeline.preprocessed_data['test_split_indices']
                else:
                    n_total = len(X_train_df)
                    test_split_indices = X_train_df.index[-int(0.2 * n_total):]
                
                available_indices = [idx for idx in test_split_indices if idx in X_train_df.index and idx in Y_train_df.index]
                sample_idx = np.random.choice(available_indices)
                
                row = X_train_df.loc[sample_idx]
                label = Y_train_df.loc[sample_idx]
                
                text = f"{row.get('designation', '')} {row.get('description', '')}".strip()
                image_file = f"image_{row['imageid']}_product_{row['productid']}.jpg"
                image_path = TRAIN_IMAGES_DIR / image_file
                
                if image_path.exists() and len(text) > 10:
                    st.session_state.test_example = {
                        'text': text,
                        'image_path': str(image_path),
                        'class_name': pipeline.category_names.get(label['prdtypecode'], 'Unknown'),
                        'class_code': label['prdtypecode'],
                        'imageid': row['imageid'],
                        'productid': row['productid'],
                        'mode': 'test_split'
                    }
                else:
                    st.error("Exemple non valide, réessayez")
                    
            except Exception as e:
                st.error(f"Erreur génération exemple: {e}")
        
        # Afficher l'exemple actuel
        if 'test_example' not in st.session_state:
            # Exemple par défaut
            st.session_state.test_example = {
                'text': "Console de jeu PlayStation 5 dernière génération avec écran OLED",
                'image_path': None,
                'class_name': 'Exemple',
                'mode': 'manual'
            }
        
        if st.session_state.test_example['mode'] == 'test_split':
            example = st.session_state.test_example
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Texte du produit:**")
                text_input = st.text_area("Description + Désignation", 
                                         value=example['text'], height=120, key="text_test_split")
                st.success(f"**Classe réelle:** {example['class_name']} ({example['class_code']})")
            
            with col2:
                st.write("**Image du produit:**")
                if example['image_path'] and os.path.exists(example['image_path']):
                    image = Image.open(example['image_path'])
                    st.image(image, caption=f"Image ID: {example['imageid']}", use_container_width=True)
                    temp_image_path = example['image_path']
                else:
                    st.error("Image non trouvée")
                    temp_image_path = None
        else:
            text_input = st.text_input("Texte du produit", "Console de jeu PlayStation 5", key="text_default")
            temp_image_path = None
    
    elif test_mode == "🏆 Exemple challenge":
        st.info("🏆 **Données challenge** : Vraies données de test (labels inconnus)")
        
        if st.button("🏆 Générer exemple challenge"):
            try:
                X_test_df = safe_read_csv(str(TEST_FILE))
                sample_idx = np.random.choice(X_test_df.index)
                row = X_test_df.loc[sample_idx]
                
                text = f"{row.get('designation', '')} {row.get('description', '')}".strip()
                image_file = f"image_{row['imageid']}_product_{row['productid']}.jpg"
                image_path = TEST_IMAGES_DIR / image_file
                
                if image_path.exists() and len(text) > 10:
                    st.session_state.test_example = {
                        'text': text,
                        'image_path': str(image_path),
                        'imageid': row['imageid'],
                        'productid': row['productid'],
                        'mode': 'challenge'
                    }
                else:
                    st.error("Exemple non valide, réessayez")
            except Exception as e:
                st.error(f"Erreur génération exemple challenge: {e}")
        
        if 'test_example' in st.session_state and st.session_state.test_example['mode'] == 'challenge':
            example = st.session_state.test_example
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Texte du produit:**")
                text_input = st.text_area("Description + Désignation", 
                                         value=example['text'], height=120, key="text_challenge")
            
            with col2:
                st.write("**Image du produit:**")
                if example['image_path'] and os.path.exists(example['image_path']):
                    image = Image.open(example['image_path'])
                    st.image(image, caption=f"Image ID: {example['imageid']}", use_container_width=True)
                    temp_image_path = example['image_path']
                else:
                    st.error("Image non trouvée")
                    temp_image_path = None
        else:
            text_input = st.text_input("Texte du produit", "Console de jeu PlayStation 5", key="text_challenge_default")
            temp_image_path = None
    
    else:  # Saisie manuelle
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Texte du produit:**")
            # Initialisation du texte par défaut
            default_text = "Smartphone Samsung Galaxy dernière génération avec écran OLED"
            
            # Si une image a été sélectionnée et qu'on a extrait le texte
            if 'extracted_text' in st.session_state:
                default_text = st.session_state.extracted_text
            
            text_input = st.text_area("Description + Désignation", 
                                     value=default_text, height=120, key="text_manual")
        
        with col2:
            st.write("**Image du produit:**")
            
            # Indiquer le chemin par défaut
            st.info("💡 Sélectionnez une image du dossier `image_test` pour détecter automatiquement le texte")
            
            uploaded_file = st.file_uploader(
                "Choisir une image du dossier image_test...", 
                type=["jpg", "jpeg", "png"],
                help="Naviguez vers le dossier data/images/image_test pour sélectionner une image"
            )
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Image sélectionnée", use_container_width=True)
                
                # Sauvegarder temporairement
                temp_dir = APP_DIR / "temp_uploads"
                temp_dir.mkdir(exist_ok=True)
                temp_image_path = temp_dir / uploaded_file.name
                image.save(str(temp_image_path))
                
                # Essayer d'extraire les infos automatiquement
                extracted_info = extract_info_from_image(uploaded_file.name)
                if extracted_info:
                    st.success("✅ Texte détecté automatiquement !")
                    st.session_state.extracted_text = extracted_info['text']
                    
                    with st.expander("📋 Infos extraites"):
                        st.write(f"**Image ID:** {extracted_info['imageid']}")
                        st.write(f"**Product ID:** {extracted_info['productid']}")
                        st.write(f"**Désignation:** {extracted_info['designation']}")
                        st.write(f"**Description:** {extracted_info['description']}")
                    
                    # Rerun pour mettre à jour le text_area
                    st.rerun()
                else:
                    st.warning("⚠️ Impossible d'extraire le texte automatiquement. Vérifiez le format du nom de fichier.")
            else:
                temp_image_path = None

    # Bouton de prédiction
    if st.button("🔍 Classifier le Produit", disabled=(temp_image_path is None)):
        try:
            with st.spinner("Classification en cours..."):
                # Charger le modèle sélectionné
                pipeline.load_model(model_type)
                
                # S'assurer que le texte est bien une chaîne de caractères
                text_input_clean = str(text_input).strip()
                
                # Effectuer la prédiction
                results = pipeline.predict_multimodal(text_input_clean, temp_image_path, fusion_strategy)
                
                # Afficher les résultats
                st.success("✅ Classification terminée!")
                
                # Résultat principal
                st.subheader("🎯 Résultat Principal")
                st.success(f"**Classe prédite:** {results['predicted_class_name']}")
                st.info(f"**Code classe:** {results['predicted_class']}")
                
                # Afficher la classe réelle si c'est un exemple test_split
                if test_mode == "🎲 Exemple test_split" and 'test_example' in st.session_state:
                    example = st.session_state.test_example
                    if example['mode'] == 'test_split':
                        is_correct = results['predicted_class'] == example['class_code']
                        if is_correct:
                            st.success(f"✅ **Correct !** Classe réelle: {example['class_name']}")
                        else:
                            st.error(f"❌ **Erreur !** Classe réelle: {example['class_name']}")
                
                # Probabilités top 5
                st.subheader("📊 Top 5 des Probabilités")
                top_indices = np.argsort(results['probabilities'])[-5:][::-1]
                top_probs = results['probabilities'][top_indices]
                top_classes = [pipeline.category_names[pipeline.idx_to_category[idx]] for idx in top_indices]
                
                prob_df = pd.DataFrame({
                    "Classe": top_classes,
                    "Probabilité": top_probs
                })
                
                # Graphique des probabilités
                fig = px.bar(prob_df, x="Probabilité", y="Classe", orientation='h',
                            title="Top 5 des Prédictions")
                st.plotly_chart(fig, use_container_width=True)
                
                # Comparaison des modalités
                st.subheader("🔄 Comparaison par Modalité")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Texte seul", 
                             pipeline.category_names[results['text_prediction']])
                
                with col2:
                    st.metric("Image seule", 
                             pipeline.category_names[results['image_prediction']])
                
                with col3:
                    st.metric("Fusion", 
                             results['predicted_class_name'])
                
        except Exception as e:
            st.error(f"❌ Erreur lors de la classification: {str(e)}")
            st.error("Vérifiez que tous les modèles sont correctement chargés.")
            
            # Informations de débogage
            with st.expander("🔍 Informations de débogage"):
                st.write(f"**Type de text_input:** {type(text_input)}")
                st.write(f"**Contenu de text_input:** {text_input}")
                st.write(f"**Chemin image:** {temp_image_path}")
                st.write(f"**Modèle sélectionné:** {model_type}")
                st.write(f"**Stratégie de fusion:** {fusion_strategy}")
                st.write(f"**Mode de test:** {test_mode}")

# ==================== PAGE EXPLICABILITÉ ====================
elif page == "🎯 Explicabilité":
    st.title("🎯 Explicabilité des Modèles")
    st.markdown("---")
    
    # Explication des deux types d'explicabilité
    with st.expander("📖 Types d'explicabilité disponibles"):
        st.markdown("""
        **📊 SHAP XGBoost** : Explique comment le modèle XGBoost utilise les features ResNet50 (embeddings 2048D) pour classifier les images.
        
        **🔗 Explicabilité Multimodale** : Compare et explique la fusion entre le modèle texte (SVM) et le modèle image (XGBoost/Neural Net).
        """)
    
    # Tabs pour séparer les deux types d'explicabilité
    tab1, tab2 = st.tabs(["📊 SHAP XGBoost (Features Images)", "🔗 Explicabilité Multimodale (Fusion)"])
    
    # ==================== TAB 1: SHAP XGBOOST ====================
    with tab1:
        st.subheader("📊 Analyses SHAP - Modèle XGBoost sur Features ResNet50")
        st.info("💡 **SHAP explique comment XGBoost** utilise les 2048 features extraites par ResNet50 pour classifier les images")
        
        # Vérifier l'existence des fichiers SHAP
        shap_reports_dir = DATA_DIR / "reports"
        shap_files = {
            "Bar Plot": shap_reports_dir / "shap_bar_plot.png",
            "Dot Plot": shap_reports_dir / "shap_dot_plot.png",
            "Waterfall Exemple 1": shap_reports_dir / "shap_waterfall_exemple_1.png",
            "Waterfall Exemple 2": shap_reports_dir / "shap_waterfall_exemple_2.png", 
            "Waterfall Exemple 3": shap_reports_dir / "shap_waterfall_exemple_3.png",
            "Force Plot Exemple 1": shap_reports_dir / "shap_force_exemple_1.png",
            "Force Plot Exemple 2": shap_reports_dir / "shap_force_exemple_2.png",
            "Force Plot Exemple 3": shap_reports_dir / "shap_force_exemple_3.png"
        }
        
        # Organiser les graphiques
        available_files = {name: path for name, path in shap_files.items() if path.exists()}
        
        if available_files:
            # Séparer les graphiques agrégés des exemples individuels
            aggregate_plots = {k: v for k, v in available_files.items() if k in ["Bar Plot", "Dot Plot"]}
            individual_plots = {k: v for k, v in available_files.items() if k not in ["Bar Plot", "Dot Plot"]}
            
            # Sub-tabs pour organiser l'affichage SHAP
            shap_tab1, shap_tab2 = st.tabs(["📈 Graphiques Agrégés", "🔍 Exemples Individuels"])
            
            with shap_tab1:
                st.info("📈 **Importance globale** : Quelles features ResNet50 XGBoost considère comme les plus importantes")
                for name, path in aggregate_plots.items():
                    st.subheader(f"📊 {name}")
                    try:
                        image = Image.open(path)
                        st.image(image, caption=f"SHAP {name} - XGBoost sur features ResNet50", use_container_width=True)
                    except Exception as e:
                        st.error(f"Erreur chargement {name}: {e}")
            
            with shap_tab2:
                st.info("🔍 **Explications individuelles** : Comment XGBoost prend sa décision pour des images spécifiques")
                # Grouper par type
                waterfall_plots = {k: v for k, v in individual_plots.items() if "Waterfall" in k}
                force_plots = {k: v for k, v in individual_plots.items() if "Force" in k}
                
                if waterfall_plots:
                    st.subheader("🌊 Waterfall Plots (Contribution de chaque feature)")
                    cols = st.columns(min(3, len(waterfall_plots)))
                    for i, (name, path) in enumerate(waterfall_plots.items()):
                        with cols[i % 3]:
                            try:
                                image = Image.open(path)
                                st.image(image, caption=name, use_container_width=True)
                            except Exception as e:
                                st.error(f"Erreur {name}: {e}")
                
                if force_plots:
                    st.subheader("⚡ Force Plots (Vue d'ensemble des contributions)") 
                    cols = st.columns(min(3, len(force_plots)))
                    for i, (name, path) in enumerate(force_plots.items()):
                        with cols[i % 3]:
                            try:
                                image = Image.open(path)
                                st.image(image, caption=name, use_container_width=True)
                            except Exception as e:
                                st.error(f"Erreur {name}: {e}")
        else:
            st.warning("⚠️ Aucune image SHAP trouvée. Exécutez d'abord l'analyse SHAP dans main.py pour générer les graphiques.")
            st.info("💡 Les fichiers SHAP devraient se trouver dans le dossier `data/reports/`")
    
    # ==================== TAB 2: EXPLICABILITÉ MULTIMODALE ====================
    with tab2:
        st.subheader("🔗 Explicabilité Multimodale - Fusion Texte + Image")
        st.info("💡 **Compare et explique** comment la fusion entre modèle texte (SVM) et modèle image (XGBoost/Neural Net) prend ses décisions")
        
        # Fonction pour récupérer des exemples test_split
        @st.cache_data
        def get_test_split_examples():
            """Récupère des exemples depuis les données test_split pour l'explicabilité"""
            try:
                # Charger les données originales
                X_train_df = safe_read_csv(str(X_TRAIN_FILE))
                Y_train_df = safe_read_csv(str(Y_TRAIN_FILE))
                
                # Récupérer les indices du test_split depuis le pipeline
                if hasattr(pipeline, 'preprocessed_data') and 'test_split_indices' in pipeline.preprocessed_data:
                    test_split_indices = pipeline.preprocessed_data['test_split_indices']
                else:
                    # Fallback
                    n_total = len(X_train_df)
                    test_split_indices = X_train_df.index[-int(0.2 * n_total):]
                    st.warning("⚠️ Indices test_split non trouvés, utilisation d'une approximation")
                
                # Prendre quelques exemples pour l'explicabilité
                available_indices = [idx for idx in test_split_indices if idx in X_train_df.index and idx in Y_train_df.index]
                sample_indices = np.random.choice(available_indices, size=min(10, len(available_indices)), replace=False)
                
                samples = []
                for idx in sample_indices:
                    row = X_train_df.loc[idx]
                    label = Y_train_df.loc[idx]
                    
                    # Construire le texte
                    text = f"{row.get('designation', '')} {row.get('description', '')}".strip()
                    
                    # Construire le chemin image
                    image_file = f"image_{row['imageid']}_product_{row['productid']}.jpg"
                    image_path = TRAIN_IMAGES_DIR / image_file
                    
                    # Nom de la classe
                    class_name = pipeline.category_names.get(label['prdtypecode'], 'Unknown')
                    
                    if image_path.exists() and len(text) > 10:
                        samples.append({
                            'text': text,
                            'image_path': image_path,
                            'class_name': class_name,
                            'class_code': label['prdtypecode'],
                            'imageid': row['imageid'],
                            'productid': row['productid'],
                            'index': idx
                        })
                
                return samples
                
            except Exception as e:
                st.error(f"❌ Erreur chargement exemples test_split: {e}")
                return []
        
        # Charger les exemples test_split
        test_examples = get_test_split_examples()
        
        # Interface pour charger un exemple
        st.subheader("📝 Sélection de l'Exemple pour Fusion")
        
        # Choix du mode
        mode = st.radio("Source des données", 
                       ["🎲 Exemple test_split", "✍️ Saisie manuelle"], key="fusion_mode")
        
        if mode == "🎲 Exemple test_split" and test_examples:
            st.info("📊 **Utilisation des données test_split** (non vues pendant l'entraînement)")
            
            # Sélectionner un exemple
            if st.button("🎲 Générer un exemple test_split", key="fusion_generate"):
                st.session_state.current_explainer_example = np.random.choice(test_examples)
            
            # Afficher l'exemple actuel
            if 'current_explainer_example' not in st.session_state and test_examples:
                st.session_state.current_explainer_example = test_examples[0]
            
            if 'current_explainer_example' in st.session_state:
                example = st.session_state.current_explainer_example
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Texte du produit:**")
                    text_input = st.text_area("Description + Désignation", 
                                             value=example['text'][:400] + "..." if len(example['text']) > 400 else example['text'],
                                             height=120, key="fusion_text")
                    st.success(f"**Classe réelle:** {example['class_name']} ({example['class_code']})")
                
                with col2:
                    st.write("**Image du produit:**")
                    if os.path.exists(example['image_path']):
                        image = Image.open(example['image_path'])
                        st.image(image, caption=f"Image ID: {example['imageid']}", use_container_width=True)
                        temp_image_path = example['image_path']
                    else:
                        st.error("Image non trouvée")
                        temp_image_path = None
        
        else:  # Saisie manuelle
            col1, col2 = st.columns(2)
            
            with col1:
                text_input = st.text_area("Texte", "Console de jeu PlayStation 5 dernière génération", key="fusion_manual_text")
            
            with col2:
                uploaded_file = st.file_uploader("Image", type=["jpg", "jpeg", "png"], key="fusion_upload")
                
                if uploaded_file is not None:
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Image pour analyse", use_container_width=True)
                    
                    temp_dir = "temp_uploads"
                    os.makedirs(temp_dir, exist_ok=True)
                    temp_image_path = os.path.join(temp_dir, uploaded_file.name)
                    image.save(temp_image_path)
                else:
                    temp_image_path = None
        
        fusion_strategy = st.selectbox("Stratégie de fusion", ["mean", "product", "weighted", "confidence_weighted"], key="fusion_strategy")
        
        if st.button("🔍 Générer les Explications Multimodales", disabled=(temp_image_path is None), key="generate_fusion_explanations"):
            try:
                with st.spinner("Génération des explications multimodales..."):
                    # S'assurer que le texte est bien une chaîne de caractères
                    def clean_text_input(text_input):
                        if isinstance(text_input, np.ndarray):
                            if text_input.size == 1:
                                return str(text_input.item()).strip()
                            else:
                                return ' '.join([str(item) for item in text_input.flatten()]).strip()
                        elif text_input is None:
                            return ""
                        else:
                            return str(text_input).strip()
                    
                    text_input_clean = clean_text_input(text_input)
                    
                    if len(text_input_clean) == 0:
                        st.error("❌ Le texte d'entrée est vide après nettoyage")
                        st.stop()
                    
                    explanations = pipeline.get_model_explanations(text_input_clean, temp_image_path, fusion_strategy)
                    
                    if 'error' in explanations:
                        st.error(f"❌ Erreur: {explanations['error']}")
                    else:
                        # Résultat de la prédiction
                        st.subheader("🎯 Résultat de la Fusion")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Classe Prédite", explanations['prediction']['predicted_class_name'])
                        with col2:
                            st.metric("Confiance", f"{explanations['prediction']['confidence']:.3f}")
                        with col3:
                            st.metric("Code Classe", explanations['prediction']['predicted_class'])
                        
                        # Comparaison avec la vérité terrain pour test_split
                        if mode == "🎲 Exemple test_split" and 'current_explainer_example' in st.session_state:
                            example = st.session_state.current_explainer_example
                            is_correct = explanations['prediction']['predicted_class'] == example['class_code']
                            
                            st.subheader("🎯 Évaluation vs Vérité Terrain")
                            eval_col1, eval_col2, eval_col3 = st.columns(3)
                            
                            with eval_col1:
                                status = "✅ Correct" if is_correct else "❌ Incorrect"
                                st.metric("Résultat", status)
                            with eval_col2:
                                st.metric("Classe Réelle", example['class_name'])
                            with eval_col3:
                                st.metric("Classe Prédite", explanations['prediction']['predicted_class_name'])
                            
                            if not is_correct:
                                st.error("🔍 **Erreur de classification détectée !**")
                        
                        # Comparaison des modalités
                        st.subheader("🔄 Comparaison des Modalités")
                        
                        ind_preds = explanations['individual_predictions']
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(
                                "📝 Texte Seul (SVM)", 
                                pipeline.category_names.get(ind_preds['text_prediction'], 'Unknown'),
                                f"Confiance: {ind_preds['text_confidence']:.3f}"
                            )
                        
                        with col2:
                            st.metric(
                                "🖼️ Image Seule (XGBoost/Neural)", 
                                pipeline.category_names.get(ind_preds['image_prediction'], 'Unknown'),
                                f"Confiance: {ind_preds['image_confidence']:.3f}"
                            )
                        
                        # Importance des modalités
                        st.subheader("⚖️ Poids des Modalités dans la Fusion")
                        
                        modality_imp = explanations['modality_importance']
                        
                        importance_df = pd.DataFrame({
                            'Modalité': ['Texte (SVM)', 'Image (XGBoost/Neural)'],
                            'Poids': [modality_imp['text_weight']*100, modality_imp['image_weight']*100]
                        })
                        
                        fig = px.pie(importance_df, values='Poids', names='Modalité',
                                    title=f"Contribution des Modalités (Stratégie: {fusion_strategy})")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Analyse détaillée
                        st.subheader("🔍 Analyse Détaillée de la Fusion")
                        
                        tabs = st.tabs(["📝 Analyse Texte", "🖼️ Analyse Image", "🔗 Analyse Fusion"])
                        
                        with tabs[0]:
                            text_analysis = explanations['text_analysis']
                            st.write("**Statistiques du texte (analysé par SVM):**")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Longueur", text_analysis['text_length'])
                            with col2:
                                st.metric("Nombre de mots", text_analysis['word_count'])
                            with col3:
                                st.metric("Confiance SVM", f"{text_analysis['text_confidence']:.3f}")
                            
                            st.write("**Premiers mots détectés:**")
                            st.write(" • ".join(text_analysis['top_words']))
                        
                        with tabs[1]:
                            image_analysis = explanations['image_analysis']
                            st.write("**Analyse des features image (par XGBoost/Neural Net):**")
                            if image_analysis['feature_importance_available']:
                                st.write("**Features ResNet50 les plus importantes (indices):**")
                                for i, feature_idx in enumerate(image_analysis['top_features'], 1):
                                    st.write(f"{i}. Feature {feature_idx}")
                                st.info("💡 Ces indices correspondent aux neurones ResNet50 les plus influents")
                            else:
                                st.info("Importance des features non disponible pour ce modèle")
                        
                        with tabs[2]:
                            fusion_analysis = explanations['fusion_analysis']
                            st.write("**Analyse de la stratégie de fusion:**")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                agreement_status = "✅ Oui" if fusion_analysis['agreement'] else "❌ Non"
                                st.metric("Accord Texte-Image", agreement_status)
                            
                            with col2:
                                boost_status = "✅ Oui" if fusion_analysis['confidence_boost'] else "❌ Non"
                                st.metric("Fusion Améliore", boost_status)
                            
                            with col3:
                                dominant = "📝 Texte" if fusion_analysis['dominant_modality'] == 'text' else "🖼️ Image"
                                st.metric("Modalité Dominante", dominant)
                            
                            # Interprétation
                            st.write("**Interprétation de la fusion:**")
                            if fusion_analysis['agreement']:
                                st.success("🎯 Les modalités texte (SVM) et image (XGBoost/Neural) sont d'accord, la prédiction est fiable")
                            else:
                                st.warning("⚠️ Désaccord entre les modalités, vérifier la prédiction de fusion")
                            
                            if fusion_analysis['confidence_boost']:
                                st.info("📈 La fusion multimodale améliore la confiance par rapport aux prédictions individuelles")
                            else:
                                st.info("📊 La fusion n'améliore pas la confiance, une modalité domine probablement")
                    
            except Exception as e:
                st.error(f"❌ Erreur génération explications multimodales: {str(e)}")
                st.info("💡 Vérifiez que tous les modèles (texte SVM + image XGBoost/Neural Net) sont correctement chargés.")
        
        # Aide
        if not test_examples and mode == "🎲 Exemple test_split":
            st.warning("⚠️ Aucun exemple test_split disponible. Utilisez le mode 'Saisie manuelle'.")
        
        # Statistiques sur les catégories (si disponible)
        if test_examples:
            with st.expander("📊 Statistiques détaillées des exemples test_split"):
                category_stats = {}
                for example in test_examples:
                    cat = example['class_name']
                    if cat not in category_stats:
                        category_stats[cat] = {'count': 0}
                    category_stats[cat]['count'] += 1
                
                st.write("**Distribution par catégorie dans les exemples:**")
                for cat_name, stats in sorted(category_stats.items(), key=lambda x: x[1]['count'], reverse=True):
                    st.write(f"• **{cat_name}**: {stats['count']} exemples")
                
                # Recommandations
                under_represented = [cat for cat, stats in category_stats.items() if stats['count'] < 2]
                if under_represented:
                    st.warning(f"⚠️ Catégories sous-représentées ({len(under_represented)}): {', '.join(under_represented[:5])}{'...' if len(under_represented) > 5 else ''}")

# Footer
st.markdown("---")
st.markdown("🛍️ **Challenge Rakuten** - Classification Multimodale | Développé avec Streamlit")