import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import main 

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
        from preprocess import ProductClassificationPipeline, PipelineConfig
        
        config = PipelineConfig.from_yaml('config.yaml')
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
        'image_models': 'data/results/image_models_comparison_results.csv',
        'text_model': 'data/results/text_model_results.csv',
        'multimodal': 'data/reports/multimodal_comparison_results.csv'
    }
    
    for key, file_path in results_files.items():
        if os.path.exists(file_path):
            try:
                results[key] = pd.read_csv(file_path, index_col=0)
            except Exception as e:
                st.warning(f"Erreur lecture {file_path}: {e}")
    
    # Rapports détaillés
    rapport_files = {
        'rapport_xgboost': 'data/rapports/rapport_xgboost.csv',
        'rapport_neural_net': 'data/rapports/rapport_neural_net.csv',
        'rapport_text_SVM': 'data/rapports/rapport_text_SVM.csv',
        'erreurs_xgboost': 'data/erreurs/erreurs_xgboost.csv',
        'erreurs_neural_net': 'data/erreurs/erreurs_neural_net.csv',
        'erreurs_text_SVM': 'data/erreurs/erreurs_text_SVM.csv'
    }
    
    for key, file_path in rapport_files.items():
        if os.path.exists(file_path):
            try:
                results[key] = pd.read_csv(file_path)
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
    
    # Mapping des codes vers les noms (basé sur votre preprocess.py)
    category_names = {
        10: "Livres", 2280: "Jeux vidéo", 50: "Jouets & Jeux",
        1280: "Accessoires téléphones", 2705: "Accessoires console",
        2522: "Équipement bébé", 2582: "Matériel & accessoires",
        1560: "Photos", 1281: "Téléphonie fixe",
        1920: "Musique amplifiée", 2403: "Livres en langues étrangères",
        1140: "TV", 2583: "Articles sport", 1180: "Décoration",
        1300: "Jeux vidéo ancien", 2462: "Fournitures bureau",
        1160: "Électroménager", 2060: "Articles soins",
        40: "DVD & Films", 60: "Consoles", 1320: "CD",
        1302: "Jeux vidéo rétro", 2220: "Puériculture",
        2905: "Instruments musique", 2585: "Sports & Loisirs",
        1940: "Instrument musique", 1301: "Consoles rétro"
    }
    
    # Mapping détaillé avec descriptions et emoji
    def get_category_description(code):
        descriptions = {
            10: {"name": "Livres", "emoji": "📚", "desc": "Romans, essais, BD, magazines"},
            40: {"name": "DVD & Films", "emoji": "🎬", "desc": "Films, séries, documentaires"},
            50: {"name": "Jouets & Jeux", "emoji": "🧸", "desc": "Jouets enfants, jeux de société"},
            60: {"name": "Consoles", "emoji": "🎮", "desc": "PlayStation, Xbox, Nintendo"},
            1140: {"name": "TV", "emoji": "📺", "desc": "Télévisions, écrans, projecteurs"},
            1160: {"name": "Électroménager", "emoji": "🏠", "desc": "Frigo, lave-linge, micro-ondes"},
            1180: {"name": "Décoration", "emoji": "🖼️", "desc": "Meubles, luminaires, objets déco"},
            1280: {"name": "Accessoires téléphones", "emoji": "📱", "desc": "Coques, chargeurs, écouteurs"},
            1281: {"name": "Téléphonie fixe", "emoji": "☎️", "desc": "Téléphones fixes, répondeurs"},
            1300: {"name": "Jeux vidéo ancien", "emoji": "🕹️", "desc": "Jeux rétro, collectors"},
            1301: {"name": "Consoles rétro", "emoji": "👾", "desc": "Anciennes consoles de jeu"},
            1302: {"name": "Jeux vidéo rétro", "emoji": "🎯", "desc": "Jeux vintage, cartouches"},
            1320: {"name": "CD", "emoji": "💿", "desc": "Musique, albums, compilations"},
            1560: {"name": "Photos", "emoji": "📷", "desc": "Appareils photo, objectifs"},
            1920: {"name": "Musique amplifiée", "emoji": "🎵", "desc": "Enceintes, amplis, sono"},
            1940: {"name": "Instrument musique", "emoji": "🎸", "desc": "Guitares, pianos, batteries"},
            2060: {"name": "Articles soins", "emoji": "🧴", "desc": "Cosmétiques, hygiène, bien-être"},
            2220: {"name": "Puériculture", "emoji": "👶", "desc": "Poussettes, biberons, vêtements bébé"},
            2280: {"name": "Jeux vidéo", "emoji": "🎮", "desc": "Jeux récents, dernières sorties"},
            2403: {"name": "Livres en langues étrangères", "emoji": "📖", "desc": "Livres anglais, multilingues"},
            2462: {"name": "Fournitures bureau", "emoji": "✏️", "desc": "Stylos, cahiers, classeurs"},
            2522: {"name": "Équipement bébé", "emoji": "🍼", "desc": "Mobilier, sécurité, éveil bébé"},
            2582: {"name": "Matériel & accessoires", "emoji": "🔧", "desc": "Outils, bricolage, jardinage"},
            2583: {"name": "Articles sport", "emoji": "⚽", "desc": "Équipement sportif, vêtements"},
            2585: {"name": "Sports & Loisirs", "emoji": "🏃", "desc": "Fitness, outdoor, loisirs créatifs"},
            2705: {"name": "Accessoires console", "emoji": "🎮", "desc": "Manettes, casques gaming"},
            2905: {"name": "Instruments musique", "emoji": "🎺", "desc": "Instruments à vent, cordes, percussions"}
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
            X_train_df = pd.read_csv('data/X_train_update.csv', index_col=0)
            Y_train_df = pd.read_csv('data/Y_train_CVw08PX.csv', index_col=0)
            
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
                    image_path = os.path.join('data/images/image_train', image_file)
                    if os.path.exists(image_path):
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
                image_path = os.path.join('data/images/image_train', image_file)
                
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
        'Classe': [10, 40, 50, 60, 1140, 1160, 1180, 1280, 1281, 1300, 1301, 1302, 1320, 1560, 1920, 1940, 2060, 2220, 2280, 2403, 2462, 2522, 2582, 2583, 2585, 2705, 2905],
        'Nom': ['Livres', 'DVD & Films', 'Jouets & Jeux', 'Consoles', 'TV', 'Électroménager', 'Décoration', 'Accessoires téléphones', 'Téléphonie fixe', 'Jeux vidéo ancien', 'Consoles rétro', 'Jeux vidéo rétro', 'CD', 'Photos', 'Musique amplifiée', 'Instrument musique', 'Articles soins', 'Puériculture', 'Jeux vidéo', 'Livres en langues étrangères', 'Fournitures bureau', 'Équipement bébé', 'Matériel & accessoires', 'Articles sport', 'Sports & Loisirs', 'Accessoires console', 'Instruments musique'],
        'Échantillons': [3116, 2508, 1681, 832, 2671, 3953, 764, 4870, 2070, 5045, 807, 2491, 3241, 5073, 4303, 803, 4993, 824, 4760, 4774, 1421, 4989, 2589, 10209, 2496, 2761, 872],
        'Pourcentage': [3.67, 2.95, 1.98, 0.98, 3.15, 4.66, 0.90, 5.74, 2.44, 5.94, 0.95, 2.93, 3.82, 5.97, 5.07, 0.95, 5.88, 0.97, 5.61, 5.62, 1.67, 5.88, 3.05, 12.02, 2.94, 3.25, 1.03]
    }
    
    df_classes = pd.DataFrame(class_data)
    
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
        comparison_data = []
        
        # Modèles image
        for model_name, row in results_data['image_models'].iterrows():
            comparison_data.append({
                'Modèle': model_name,
                'Type': 'Image',
                'Accuracy': row['accuracy'],
                'F1-Score': row['weighted_f1'],
                'Précision': row.get('weighted_precision', 0),
                'Rappel': row.get('weighted_recall', 0)
            })
        
        # Modèle texte
        for model_name, row in results_data['text_model'].iterrows():
            comparison_data.append({
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
                comparison_data.append({
                    'Modèle': model_name,
                    'Type': model_type,
                    'Accuracy': row['accuracy'],
                    'F1-Score': row['weighted_f1'],
                    'Précision': row.get('weighted_precision', 0),
                    'Rappel': row.get('weighted_recall', 0)
                })
        
        df_comparison = pd.DataFrame(comparison_data)
        
        if not df_comparison.empty:
            # Graphique de comparaison
            fig = px.scatter(df_comparison, x='Accuracy', y='F1-Score', 
                            color='Type', size='Précision',
                            hover_data=['Modèle', 'Rappel'],
                            title="Performance des Modèles (Accuracy vs F1-Score)")
            st.plotly_chart(fig, use_container_width=True)
            
            # Tableau récapitulatif
            st.subheader("📋 Tableau Récapitulatif")
            df_display = df_comparison.round(3)
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
                        elif 'prediction_probability' in error.columns:
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
            comparison_fig = px.bar(
                x=multimodal_results.index,
                y=multimodal_results['weighted_f1'],
                color=multimodal_results['model_type'],
                title="Comparaison Individuel vs Multimodal"
            )
            st.plotly_chart(comparison_fig, use_container_width=True)
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
    
    with col2:
        fusion_strategy = st.selectbox("Stratégie de Fusion", 
                                      ["mean", "product", "weighted", "confidence_weighted"])
    
    # Interface de test
    st.subheader("📝 Saisie des Données")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Texte du produit:**")
        text_input = st.text_area("Description + Désignation", 
                                 value="Smartphone Samsung Galaxy dernière génération avec écran OLED",
                                 height=100)
    
    with col2:
        st.write("**Image du produit:**")
        uploaded_file = st.file_uploader("Choisir une image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Image téléchargée", use_container_width=True)
            
            # Sauvegarder temporairement
            temp_dir = "temp_uploads"
            os.makedirs(temp_dir, exist_ok=True)
            temp_image_path = os.path.join(temp_dir, uploaded_file.name)
            image.save(temp_image_path)
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

# ==================== PAGE EXPLICABILITÉ ====================
elif page == "🎯 Explicabilité":
    st.title("🎯 Explicabilité des Modèles")
    st.markdown("---")
    
    st.info("💡 Cette section fournit des explications simplifiées sur les prédictions multimodales en utilisant les données test_split.")
    
    # Fonction pour récupérer des exemples test_split
    @st.cache_data
    def get_test_split_examples():
        """Récupère des exemples depuis les données test_split pour l'explicabilité"""
        try:
            # Charger les données originales
            X_train_df = pd.read_csv('data/X_train_update.csv', index_col=0)
            Y_train_df = pd.read_csv('data/Y_train_CVw08PX.csv', index_col=0)
            
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
                image_path = os.path.join('data/images/image_train', image_file)
                
                # Nom de la classe
                class_name = pipeline.category_names.get(label['prdtypecode'], 'Unknown')
                
                if os.path.exists(image_path) and len(text) > 10:
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
    st.subheader("📝 Sélection de l'Exemple")
    
    # Choix du mode
    mode = st.radio("Source des données", 
                   ["🎲 Exemple test_split", "✍️ Saisie manuelle"])
    
    if mode == "🎲 Exemple test_split" and test_examples:
        st.info("📊 **Utilisation des données test_split** (non vues pendant l'entraînement)")
        
        # Sélectionner un exemple
        if st.button("🎲 Générer un exemple test_split"):
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
                                         height=120)
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
            text_input = st.text_area("Texte", "Console de jeu PlayStation 5 dernière génération")
        
        with col2:
            uploaded_file = st.file_uploader("Image", type=["jpg", "jpeg", "png"])
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Image pour analyse", use_container_width=True)
                
                temp_dir = "temp_uploads"
                os.makedirs(temp_dir, exist_ok=True)
                temp_image_path = os.path.join(temp_dir, uploaded_file.name)
                image.save(temp_image_path)
            else:
                temp_image_path = None
    
    fusion_strategy = st.selectbox("Stratégie de fusion", ["mean", "product", "weighted", "confidence_weighted"])
    
    if st.button("🔍 Générer les Explications", disabled=(temp_image_path is None)):
        try:
            with st.spinner("Génération des explications..."):
                # S'assurer que le texte est bien une chaîne de caractères (même correction que pour les tests)
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
                    st.subheader("🎯 Résultat de la Prédiction")
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
                            "📝 Texte Seul", 
                            pipeline.category_names.get(ind_preds['text_prediction'], 'Unknown'),
                            f"Confiance: {ind_preds['text_confidence']:.3f}"
                        )
                    
                    with col2:
                        st.metric(
                            "🖼️ Image Seule", 
                            pipeline.category_names.get(ind_preds['image_prediction'], 'Unknown'),
                            f"Confiance: {ind_preds['image_confidence']:.3f}"
                        )
                    
                    # Importance des modalités
                    st.subheader("⚖️ Importance des Modalités")
                    
                    modality_imp = explanations['modality_importance']
                    
                    importance_df = pd.DataFrame({
                        'Modalité': ['Texte', 'Image'],
                        'Poids': [modality_imp['text_weight']*100, modality_imp['image_weight']*100]
                    })
                    
                    fig = px.pie(importance_df, values='Poids', names='Modalité',
                                title=f"Contribution des Modalités (Stratégie: {fusion_strategy})")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Analyse détaillée
                    st.subheader("🔍 Analyse Détaillée")
                    
                    tabs = st.tabs(["📝 Analyse Texte", "🖼️ Analyse Image", "🔗 Analyse Fusion"])
                    
                    with tabs[0]:
                        text_analysis = explanations['text_analysis']
                        st.write("**Statistiques du texte:**")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Longueur", text_analysis['text_length'])
                        with col2:
                            st.metric("Nombre de mots", text_analysis['word_count'])
                        with col3:
                            st.metric("Confiance", f"{text_analysis['text_confidence']:.3f}")
                        
                        st.write("**Premiers mots:**")
                        st.write(" • ".join(text_analysis['top_words']))
                    
                    with tabs[1]:
                        image_analysis = explanations['image_analysis']
                        if image_analysis['feature_importance_available']:
                            st.write("**Features les plus importantes (indices):**")
                            for i, feature_idx in enumerate(image_analysis['top_features'], 1):
                                st.write(f"{i}. Feature {feature_idx}")
                        else:
                            st.info("Importance des features non disponible pour ce modèle")
                    
                    with tabs[2]:
                        fusion_analysis = explanations['fusion_analysis']
                        st.write("**Analyse de la fusion:**")
                        
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
                        st.write("**Interprétation:**")
                        if fusion_analysis['agreement']:
                            st.success("🎯 Les deux modalités sont d'accord, la prédiction est fiable")
                        else:
                            st.warning("⚠️ Désaccord entre modalités, vérifier la prédiction")
                        
                        if fusion_analysis['confidence_boost']:
                            st.info("📈 La fusion améliore la confiance de la prédiction")
                        else:
                            st.info("📊 La fusion n'améliore pas la confiance")
                
        except Exception as e:
            st.error(f"❌ Erreur génération explications: {str(e)}")
            st.info("💡 Vérifiez que tous les modèles sont correctement chargés.")
    
    # Aide
    if not test_examples and mode == "🎲 Exemple test_split":
        st.warning("⚠️ Aucun exemple test_split disponible. Utilisez le mode 'Saisie manuelle'.")
    
    # Suggestions d'amélioration
    if len(test_examples) < 10:
        with st.expander("💡 Suggestions d'amélioration"):
            st.markdown("""
            **Problème identifié**: Peu d'exemples test_split disponibles avec images.
            
            **Causes possibles**:
            - Images manquantes dans le dossier `data/images/image_train/`
            - Test split trop petit ou mal réparti
            - Problèmes de chemins vers les images
            
            **Solutions suggérées**:
            1. Vérifier l'intégrité du dataset d'images
            2. Augmenter la taille du test_split (actuellement 20%)
            3. Équilibrer la répartition des catégories dans le split
            4. Utiliser des exemples synthétiques pour les catégories sous-représentées
            """)
    
    # Statistiques sur les catégories
    if test_examples:
        with st.expander("📊 Statistiques détaillées"):
            category_stats = {}
            for example in test_examples:
                cat = example['class_name']
                if cat not in category_stats:
                    category_stats[cat] = {
                        'count': 0, 
                        'emoji': example['class_emoji'],
                        'desc': example['class_desc']
                    }
                category_stats[cat]['count'] += 1
            
            st.write("**Distribution par catégorie**:")
            for cat_name, stats in sorted(category_stats.items(), key=lambda x: x[1]['count'], reverse=True):
                st.write(f"{stats['emoji']} **{cat_name}**: {stats['count']} exemples - {stats['desc']}")
            
            # Recommandations
            under_represented = [cat for cat, stats in category_stats.items() if stats['count'] < 2]
            if under_represented:
                st.warning(f"⚠️ Catégories sous-représentées ({len(under_represented)}): {', '.join(under_represented[:5])}{'...' if len(under_represented) > 5 else ''}")

# Footer
st.markdown("---")
st.markdown("🛍️ **Challenge Rakuten** - Classification Multimodale | Développé avec Streamlit")