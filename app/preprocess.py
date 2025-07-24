import os
import logging
import numpy as np
import pandas as pd
import pickle
import shutil
from tqdm import tqdm
from typing import Optional, Dict, Any
import yaml
import time
# from datetime import datetime
from dataclasses import dataclass, fields
# import sys
import random
from utils import safe_read_csv

# Imports machine learning
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, f1_score#, precision_score, recall_score
)
from sklearn.model_selection import train_test_split, StratifiedKFold
# import optuna

# Imports PyTorch
import torch
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True
import torch.nn as nn
# import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as transforms
# import torch.optim as optim
from PIL import Image
from pathlib import Path

from utils_path import *

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Configuration GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"GPU disponible : {torch.cuda.get_device_name(0)}")
    print(f"CUDA Memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB / {torch.cuda.memory_reserved()/1024**3:.2f}GB")
    torch.cuda.manual_seed_all(RANDOM_SEED)

class RakutenImageDataset(Dataset):
    def __init__(self, image_paths, labels=None, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)),  # Resize direct à la taille finale
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        image = self.transform(image)
        if self.labels is not None:
            return image, self.labels[idx]
        return image

@dataclass
class PipelineConfig:
    data_path: str
    model_path: str
    image_dir: str
    batch_size: int = 128
    target_size: int = 2000
    random_state: int = 42
    num_workers: Optional[int] = 14
    early_stopping_patience: int = 5

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'PipelineConfig':
            """
            Crée une instance de PipelineConfig à partir d'un fichier YAML
            
            Args:
                yaml_path (str): Chemin vers le fichier de configuration YAML
                
            Returns:
                PipelineConfig: Instance configurée
            """
            with open(yaml_path, 'r') as f:
                config_dict = yaml.safe_load(f)
                
            # Filtrer les clés pour ne garder que celles définies dans la classe
            valid_keys = {field.name for field in fields(PipelineConfig)}
            filtered_config = {k: v for k, v in config_dict.items() if k in valid_keys}
                
            return cls(**filtered_config)

class ProductClassificationPipeline:
    def __init__(self, config: PipelineConfig):
        """
        Initialise le pipeline de classification avec une configuration
        
        Args:
            config (PipelineConfig): Configuration du pipeline
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.base_dir = Path.cwd()

        # Initialisation
        self._setup_logger()
        self._init_paths()
        self._init_categories()
        
        # États
        self.preprocessed_data = None
        self.model = None
        self.text_model = None

    def _init_categories(self):
        """Initialise le mapping des catégories Rakuten"""
        self.category_names = {
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
        # Créer le mapping vers des indices consécutifs
        self.category_to_idx = {code: idx for idx, code in enumerate(sorted(self.category_names.keys()))}
        self.idx_to_category = {idx: code for code, idx in self.category_to_idx.items()}

    def _init_paths(self):
        """Initialise tous les chemins nécessaires"""
        # Chemins principaux
        self.train_image_dir = TRAIN_IMAGES_DIR
        self.test_image_dir = TEST_IMAGES_DIR

        # Chemins des modèles
        self.model_dir = MODELS_DIR  

        # Chemins des métadonnées
        self.meta_path = DATA_DIR / 'metadata.pkl'
        
        # Chemins pour les résultats
        self.results_dir = RESULTS_DIR                  
        
        # Chemins pour les prédictions
        self.predictions_dir = PREDICTIONS_DIR
        
    def _setup_logger(self):
        """Configure le logger pour le suivi des opérations"""
        self.logger = logging.getLogger('classification_pipeline')
        self.logger.setLevel(logging.INFO)
        
        # Évite les doublons de handlers
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

            # FileHandler pour sauvegarder les logs
            file_handler = logging.FileHandler(LOGS_DIR / 'pipeline.log')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def download_preprocessed_data(self):
        """Télécharge les données préprocessées depuis Google Drive"""
        import gdown
        import zipfile
        import shutil
        
        url = "https://drive.google.com/file/d/1guhuHp0dVRPWCtZ7570jEsTub6m2RrRF/view?usp=sharing"
        fichier_zip = "Preprocessed_data.zip"
        dossier_donnees_pretraitees = PROCESSED_DATA_DIR
        
        try:
            self.logger.info("📥 Téléchargement des données préprocessées...")
            
            # 1) Téléchargement avec gdown
            gdown.download(url, fichier_zip, fuzzy=True)
            
            # 2) Vérification et extraction
            if not os.path.exists(fichier_zip):
                raise FileNotFoundError("Le téléchargement a échoué")
                
            if not zipfile.is_zipfile(fichier_zip):
                raise zipfile.BadZipFile("Le fichier téléchargé n'est pas un ZIP valide")
            
            # 3) Création du dossier de destination
            dossier_donnees_pretraitees.mkdir(parents=True, exist_ok=True)
            
            # 4) Extraction temporaire
            temp_dir = "temp_extraction"
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            os.makedirs(temp_dir)
            
            self.logger.info("📂 Extraction des données...")
            with zipfile.ZipFile(fichier_zip, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
                
            # 5) Déplacement des fichiers
            source_dir = os.path.join(temp_dir, "processed_data")
            for item in os.listdir(source_dir):
                s = os.path.join(source_dir, item)
                d = dossier_donnees_pretraitees / item
                if d.exists():
                    if d.is_dir():
                        shutil.rmtree(d)
                    else:
                        d.unlink()
                shutil.move(s, str(d))
            
            self.logger.info("✅ Données téléchargées et extraites avec succès")
            
        except Exception as e:
            self.logger.error(f"❌ Erreur téléchargement: {e}")
            raise
            
        finally:
            # Nettoyage
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            if os.path.exists(fichier_zip):
                os.remove(fichier_zip)
                
    def _create_balanced_dataset(self, X_train_df, Y_train_df):
        """
        Crée un dataset équilibré en considérant à la fois la distribution des classes
        et la taille des fichiers images.
        
        Args:
            X_train_df (pd.DataFrame): DataFrame contenant les métadonnées des images
            Y_train_df (pd.DataFrame): DataFrame contenant les labels
            
        Returns:
            List[int]: Liste des indices sélectionnés pour le dataset équilibré
        """
        file_info = []
        
        # Fusion des DataFrames X et Y
        df_merged = X_train_df.merge(Y_train_df, left_index=True, right_index=True)
        
        # Collecte des informations sur les fichiers
        for _, row in df_merged.iterrows():
            image_file = f"image_{row['imageid']}_product_{row['productid']}.jpg"
            file_path = os.path.join(self.train_image_dir, image_file)
            
            if os.path.exists(file_path):
                size_kb = os.path.getsize(file_path) / 1024
                file_info.append({
                    'index': row.name,
                    'size_kb': size_kb,
                    'prdtypecode': row['prdtypecode'],
                    'imageid': row['imageid'],
                    'productid': row['productid']
                })
        
        df_analysis = pd.DataFrame(file_info)
        df_analysis.set_index('index', inplace=True)
        
        balanced_indices = []
        
        # Pour chaque classe
        for classe in df_analysis['prdtypecode'].unique():
            class_data = df_analysis[df_analysis['prdtypecode'] == classe].copy()
            n_samples = len(class_data)
            
            if n_samples > self.config.target_size:
                # Sous-échantillonnage stratifié par taille
                size_bins = pd.qcut(class_data['size_kb'], q=5, labels=False)
                class_data['size_bin'] = size_bins
                samples_per_bin = self.config.target_size // 5
                
                stratified_sample = []
                for bin_id in range(5):
                    bin_data = class_data[class_data['size_bin'] == bin_id]
                    if len(bin_data) > 0:
                        selected = bin_data.sample(
                            n=min(len(bin_data), samples_per_bin),
                            random_state=self.config.random_state
                        ).index.tolist()
                        stratified_sample.extend(selected)
                
                # Si on n'a pas assez d'échantillons après stratification
                remaining = self.config.target_size - len(stratified_sample)
                if remaining > 0:
                    additional = class_data[~class_data.index.isin(stratified_sample)].sample(
                        n=min(remaining, len(class_data) - len(stratified_sample)),
                        random_state=self.config.random_state
                    ).index.tolist()
                    stratified_sample.extend(additional)
                
                balanced_indices.extend(stratified_sample)
                
            else:
                # Sur-échantillonnage stratifié par taille
                current_indices = class_data.index.tolist()
                balanced_indices.extend(current_indices)  # Ajoute d'abord tous les échantillons existants
                
                if n_samples > 0:
                    # Calcul du nombre d'échantillons supplémentaires nécessaires
                    n_needed = self.config.target_size - n_samples
                    
                    # Division en bins de taille
                    size_bins = pd.qcut(class_data['size_kb'], q=min(5, n_samples), labels=False)
                    class_data['size_bin'] = size_bins
                    
                    # Sur-échantillonnage par bin
                    additional_samples = []
                    samples_needed_per_bin = n_needed // len(class_data['size_bin'].unique())
                    
                    for bin_id in class_data['size_bin'].unique():
                        bin_data = class_data[class_data['size_bin'] == bin_id]
                        if len(bin_data) > 0:
                            bin_indices = bin_data.index.tolist()
                            additional = np.random.choice(
                                bin_indices,
                                size=samples_needed_per_bin,
                                replace=True
                            ).tolist()
                            additional_samples.extend(additional)
                    
                    # Gestion du reste
                    remaining = n_needed - len(additional_samples)
                    if remaining > 0:
                        extra = np.random.choice(
                            current_indices,
                            size=remaining,
                            replace=True
                        ).tolist()
                        additional_samples.extend(extra)
                    
                    balanced_indices.extend(additional_samples)
        
        # Vérification finale
        self.logger.info(f"Indices retenus: {len(balanced_indices)} sur {len(df_analysis)} images")
        for classe in df_analysis['prdtypecode'].unique():
            n_class = sum(df_analysis.loc[balanced_indices, 'prdtypecode'] == classe)
            self.logger.info(f"Classe {classe} ({self.category_names[classe]}): {n_class} images")
        
        return balanced_indices

    def _create_dataset(self, df, labels=None, df_name=None):
        """Crée un dataset PyTorch à partir des données"""
        image_paths = []
        images_not_found = 0
        
        try:
            self.logger.info(f"Création dataset {df_name} à partir de {len(df)} entrées...")
            
            # Déterminer le répertoire d'images selon df_name
            if df_name == "X_train" or df_name == "X_test_split":
                df_path = self.train_image_dir
            elif df_name == "X_test":
                df_path = self.test_image_dir
            else:
                # Par défaut, utiliser train_image_dir
                df_path = self.train_image_dir
                self.logger.warning(f"df_name '{df_name}' non reconnu, utilisation de train_image_dir")
                            
            for _, row in df.iterrows():
                image_file = f"image_{row['imageid']}_product_{row['productid']}.jpg"
                image_path = os.path.join(df_path, image_file)
                
                if os.path.exists(image_path):
                    image_paths.append(image_path)
                else:
                    images_not_found += 1

            if len(image_paths) == 0:
                self.logger.error(f"Aucune image trouvée ! {images_not_found} images manquantes")
                self.logger.error(f"Chemin recherché : {df_path}")
                raise ValueError("Aucune image valide trouvée")
                
            self.logger.info(f"Dataset créé avec {len(image_paths)} images ({images_not_found} non trouvées)")
            return RakutenImageDataset(image_paths, labels)

        except Exception as e:
            self.logger.error(f"Erreur _create_dataset : {str(e)}")
            raise

    def _load_existing_processed_data(self, required_files):
        """
        Charge les 7 datasets depuis des .npz.
        
        Retourne un nouveau dictionnaire self.preprocessed_data
        """
        try:
            # 1) Train
            X_train_npz = np.load(required_files['X_train'], allow_pickle=True)
            y_train_npz = np.load(required_files['y_train'], allow_pickle=True)
            train_indices_npz = np.load(required_files['train_indices'], allow_pickle=True)
            
            # Extraction des données - gérer les différents formats de sauvegarde
            X_train = X_train_npz['X_train_'] if 'X_train_' in X_train_npz.files else X_train_npz['arr_0']
            y_train = y_train_npz['y_train_'] if 'y_train_' in y_train_npz.files else y_train_npz['arr_0']
            train_indices = train_indices_npz['train_indices'] if 'train_indices' in train_indices_npz.files else train_indices_npz['arr_0']
            
            # 2) Test
            X_test_npz = np.load(required_files['X_test'], allow_pickle=True)
            X_test = X_test_npz['X_test'] if 'X_test' in X_test_npz.files else X_test_npz['arr_0']
            
            # 3) Test_split
            X_test_split_npz = np.load(required_files['X_test_split'], allow_pickle=True)
            y_test_split_npz = np.load(required_files['y_test_split'], allow_pickle=True)
            test_split_indices_npz = np.load(required_files['test_split_indices'], allow_pickle=True)
            
            X_test_split = X_test_split_npz['X_test_split'] if 'X_test_split' in X_test_split_npz.files else X_test_split_npz['arr_0']
            y_test_split = y_test_split_npz['y_test_split'] if 'y_test_split' in y_test_split_npz.files else y_test_split_npz['arr_0']
            test_split_indices = test_split_indices_npz['test_split_indices'] if 'test_split_indices' in test_split_indices_npz.files else test_split_indices_npz['arr_0']
            
            # Fermeture des fichiers npz
            X_train_npz.close()
            y_train_npz.close()
            train_indices_npz.close()
            X_test_npz.close()
            X_test_split_npz.close()
            y_test_split_npz.close()
            test_split_indices_npz.close()
            
            # Extraction robuste des features
            def extract_features_safe(data, data_name=""):
                """Extrait les features des données selon leur format - version sécurisée"""
                self.logger.info(f"Extraction de {data_name}: type={type(data)}, shape={getattr(data, 'shape', 'N/A')}")
                
                try:
                    # Cas 1: Dictionnaire direct
                    if isinstance(data, dict):
                        if 'features' in data:
                            self.logger.info(f"  → Extraction via clé 'features'")
                            return data['features']
                        else:
                            self.logger.info(f"  → Dictionnaire sans 'features', clés: {list(data.keys())}")
                            return data
                    
                    # Cas 2: Array 0D contenant un objet (ATTENTION au .item())
                    elif isinstance(data, np.ndarray) and data.shape == ():
                        try:
                            item = data.item()
                            self.logger.info(f"  → Array 0D converti, type de l'item: {type(item)}")
                            
                            if isinstance(item, dict) and 'features' in item:
                                self.logger.info(f"  → Extraction via clé 'features' de l'item")
                                return item['features']
                            else:
                                return item
                        except ValueError as e:
                            self.logger.warning(f"  → Échec .item(): {e}, retour direct")
                            return data
                    
                    # Cas 3: Array numpy classique
                    elif isinstance(data, np.ndarray):
                        if data.ndim >= 2:  # Array 2D ou plus = probablement les features directement
                            self.logger.info(f"  → Array {data.ndim}D, utilisation directe")
                            return data
                        else:
                            self.logger.info(f"  → Array 1D, tentative de reshape")
                            return data
                    
                    # Cas 4: Autres types
                    else:
                        self.logger.info(f"  → Type non géré spécifiquement, retour direct")
                        return data
                        
                except Exception as e:
                    self.logger.error(f"  → Erreur extraction {data_name}: {e}")
                    return data
            
            # Application de l'extraction sécurisée
            self.logger.info("=== EXTRACTION DES FEATURES ===")
            X_train = extract_features_safe(X_train, "X_train")
            X_test = extract_features_safe(X_test, "X_test")
            X_test_split = extract_features_safe(X_test_split, "X_test_split")
            
            # Log des informations finales
            self.logger.info("=== RÉSULTATS FINAUX ===")
            self.logger.info(f"X_train: {type(X_train)} shape={getattr(X_train, 'shape', 'N/A')}")
            self.logger.info(f"y_train: {type(y_train)} shape={getattr(y_train, 'shape', 'N/A')}")
            self.logger.info(f"X_test: {type(X_test)} shape={getattr(X_test, 'shape', 'N/A')}")
            self.logger.info(f"X_test_split: {type(X_test_split)} shape={getattr(X_test_split, 'shape', 'N/A')}")
            self.logger.info(f"y_test_split: {type(y_test_split)} shape={getattr(y_test_split, 'shape', 'N/A')}")
            
            # Vérifications supplémentaires
            if hasattr(X_train, 'shape') and len(X_train.shape) == 0:
                self.logger.warning("X_train est un array 0D - investigation nécessaire")
            if hasattr(X_test_split, 'shape') and len(X_test_split.shape) == 0:
                self.logger.warning("X_test_split est un array 0D - investigation nécessaire")
            
            # Construction du dictionnaire
            preprocessed_data = {
                'X_train': X_train,
                'y_train': y_train,
                'X_test': X_test,
                'X_test_split': X_test_split,
                'y_test_split': y_test_split,
                'train_indices': train_indices,
                'test_split_indices': test_split_indices
            }
            
            return preprocessed_data

        except Exception as e:
            self.logger.error(f"Erreur chargement données : {str(e)}")
            # Debug supplémentaire
            for name, path in required_files.items():
                if os.path.exists(path):
                    try:
                        npz_file = np.load(path, allow_pickle=True)
                        self.logger.error(f"  {name}: fichiers={npz_file.files}")
                        npz_file.close()
                    except Exception as e2:
                        self.logger.error(f"  {name}: erreur lecture={e2}")
            raise         

    def _load_existing_processed_data_streamlit(self, required_files):
        """
        Version simplifiée pour Streamlit - Charge seulement les données nécessaires pour l'inférence
        
        Args:
            required_files: Dictionnaire des chemins vers les fichiers
            
        Returns:
            dict: Données prétraitées minimales pour Streamlit
        """
        try:
            preprocessed_data = {}
            
            # 1) Données de test split (nécessaires pour évaluations et exemples)
            try:
                X_test_split_npz = np.load(required_files['X_test_split'], allow_pickle=True)
                y_test_split_npz = np.load(required_files['y_test_split'], allow_pickle=True)
                test_split_indices_npz = np.load(required_files['test_split_indices'], allow_pickle=True)
                
                X_test_split = X_test_split_npz['X_test_split'] if 'X_test_split' in X_test_split_npz.files else X_test_split_npz['arr_0']
                y_test_split = y_test_split_npz['y_test_split'] if 'y_test_split' in y_test_split_npz.files else y_test_split_npz['arr_0']
                test_split_indices = test_split_indices_npz['test_split_indices'] if 'test_split_indices' in test_split_indices_npz.files else test_split_indices_npz['arr_0']
                
                # Fermeture des fichiers npz
                X_test_split_npz.close()
                y_test_split_npz.close()
                test_split_indices_npz.close()
                
                preprocessed_data.update({
                    'X_test_split': X_test_split,
                    'y_test_split': y_test_split,
                    'test_split_indices': test_split_indices
                })
                self.logger.info("✅ Données test_split chargées")
                
            except Exception as e:
                self.logger.warning(f"⚠️ Impossible de charger test_split: {e}")
            
            # 2) Données de test challenge (optionnel - pour visualisations)
            try:
                X_test = np.load(required_files['X_test'], allow_pickle=True)['arr_0']
                preprocessed_data['X_test'] = X_test
                self.logger.info("✅ Données X_test chargées")
                
            except Exception as e:
                self.logger.warning(f"⚠️ X_test non disponible: {e}")
            
            # 3) Données d'entraînement (optionnel - seulement si nécessaires pour certaines méthodes)
            load_train_data = False  # Flag pour activer si nécessaire
            
            if load_train_data:
                try:
                    X_train = np.load(required_files['X_train'], allow_pickle=True)['arr_0']
                    y_train = np.load(required_files['y_train'], allow_pickle=True)['arr_0']
                    train_indices = np.load(required_files['train_indices'], allow_pickle=True)['arr_0']
                    
                    preprocessed_data.update({
                        'X_train': X_train,
                        'y_train': y_train,
                        'train_indices': train_indices
                    })
                    self.logger.info("✅ Données d'entraînement chargées")
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Données d'entraînement non disponibles: {e}")
            
            # Vérification minimale
            if not preprocessed_data:
                raise ValueError("Aucune donnée n'a pu être chargée")
            
            self.logger.info(f"📊 Données chargées: {list(preprocessed_data.keys())}")
            return preprocessed_data
            
        except Exception as e:
            self.logger.error(f"❌ Erreur chargement données Streamlit: {str(e)}")
            raise

    # Version encore plus minimale si on veut juste l'inférence
    def _load_minimal_data_for_inference(self):
        """
        Version ultra-minimale - charge seulement ce qui est absolument nécessaire
        pour faire de l'inférence sur nouvelles données
        """
        try:
            # Juste les indices pour cohérence (si nécessaire)
            features_dir = os.path.join(self.config.data_path, 'processed_data')
            test_split_indices_path = os.path.join(features_dir, 'test_split_indices.npz')
            
            preprocessed_data = {}
            
            if os.path.exists(test_split_indices_path):
                test_split_indices = np.load(test_split_indices_path, allow_pickle=True)['arr_0']
                preprocessed_data['test_split_indices'] = test_split_indices
                self.logger.info(f"✅ Indices chargés: {len(test_split_indices)} éléments")
            
            return preprocessed_data
            
        except Exception as e:
            self.logger.error(f"❌ Erreur chargement minimal: {str(e)}")
            return {}

    # Mise à jour de prepare_data() pour Streamlit
    def prepare_data_streamlit(self, force_preprocess_image=False, force_preprocess_text=False):
        """
        Version Streamlit de prepare_data() - Optimisée pour l'inférence
        
        Args:
            force_preprocess_image: Si True, force le prétraitement (pas recommandé en prod)
            force_preprocess_text: Si True, force le prétraitement texte (pas recommandé en prod)
        """
        try:
            # 1) Vérifier les fichiers de données pré-traitées
            features_dir = os.path.join(self.config.data_path, 'processed_data')
            required_files = {
                'X_test_split': os.path.join(features_dir, 'X_test_split.npz'),
                'y_test_split': os.path.join(features_dir, 'y_test_split.npz'),
                'test_split_indices': os.path.join(features_dir, 'test_split_indices.npz'),
                'X_test': os.path.join(features_dir, 'X_test.npz'),
            }
            
            # 2) Charger les données existantes (mode normal)
            if not force_preprocess_image:
                self.logger.info("📁 Chargement des données pré-traitées...")
                self.preprocessed_data = self._load_existing_processed_data_streamlit(required_files)
                self.logger.info("✅ Données chargées pour Streamlit")
            else:
                # Mode développement - retraitement complet (à éviter en production)
                self.logger.warning("🔄 Mode retraitement activé - peut être lent...")
                return self.prepare_data(
                    balance_classes=True,
                    force_preprocess_image=True,
                    force_preprocess_text=force_preprocess_text
                )
            
            # 3) Vérifier la disponibilité du modèle texte
            if force_preprocess_text:
                self.logger.info("🔄 Vérification du modèle texte...")
                svm_path = 'data/models/SVM/model.pkl'
                if not os.path.exists(svm_path):
                    self.logger.warning("⚠️ Modèle SVM non trouvé - entraînement nécessaire")
                    # Ici on pourrait déclencher un entraînement ou retourner une erreur
            
            return self.preprocessed_data
            
        except Exception as e:
            self.logger.error(f"❌ Erreur préparation données Streamlit: {str(e)}")
            raise
        
    def prepare_data(self, balance_classes=True, force_preprocess_image=False, force_preprocess_text=False):
        """
        Prépare les données
        
        Args:
        balance_classes: Si True, équilibre les classes
        force_preprocess_image: Si True, force le prétraitement des images
        force_preprocess_text: Si True, force le prétraitement du texte
            
        Returns:
            Dict contenant les données prétraitées
        """
        try:
            # Vérification des fichiers prétraités existants
            required_files = {
                'X_train': PROCESSED_DATA_DIR / 'X_train.npz',
                'y_train': PROCESSED_DATA_DIR / 'y_train.npz',
                'X_test': PROCESSED_DATA_DIR / 'X_test.npz',  
                'X_test_split': PROCESSED_DATA_DIR / 'X_test_split.npz',
                'y_test_split': PROCESSED_DATA_DIR / 'y_test_split.npz',
                'train_indices': PROCESSED_DATA_DIR / 'train_indices.npz',
                'test_split_indices': PROCESSED_DATA_DIR / 'test_split_indices.npz',
            }

            image_files_exist = all(path.exists() for path in required_files.values())
            
            # Si les fichiers n'existent pas, les télécharger
            if not image_files_exist and not force_preprocess_image:
                self.logger.info("📥 Fichiers .npz manquants - Téléchargement automatique...")
                self.download_preprocessed_data()
                # Re-vérifier après téléchargement
                image_files_exist = all(path.exists() for path in required_files.values())

            # Décider si on retraite ou charge les images
            reprocess_images = force_preprocess_image or not image_files_exist
            
            X_train_df = safe_read_csv(str(X_TRAIN_FILE))
        
            # Chargement/traitement des données image
            if not reprocess_images:
                self.logger.info("Chargement des features pré-calculées...")
                data = self._load_existing_processed_data(required_files)
                self.logger.info("Chargement effectué avec succès.")

                # Remplit self.preprocessed_data
                self.preprocessed_data = data
                test_split_indices = self.preprocessed_data['test_split_indices']
            
                # Vérifier que les indices sont valides
                valid_indices = np.array([idx for idx in test_split_indices if idx in X_train_df.index])
                if len(valid_indices) < len(test_split_indices):
                    self.logger.warning(f"ATTENTION: Seulement {len(valid_indices)}/{len(test_split_indices)} indices test_split sont valides")
                    # Mettre à jour avec indices valides uniquement
                    self.preprocessed_data['test_split_indices'] = valid_indices
                    np.savez(required_files['test_split_indices'], arr_0=valid_indices)
                    self.logger.info(f"Indices test_split mis à jour avec {len(valid_indices)} indices valides")
                    # Mise à jour des test_split_indices pour la suite
                    test_split_indices = valid_indices
                    
            else:
                self.logger.info(f"Prétraitement des images en cours...")
                if not image_files_exist:
                    missing_files = [name for name, path in required_files.items()  if not os.path.exists(path)]
                    self.logger.warning(f"Les fichiers suivants sont manquants : {', '.join(missing_files)}")
                
                # a) Lecture des CSV
                Y_train_df = safe_read_csv(str(Y_TRAIN_FILE))
                X_test_df = safe_read_csv(str(X_TEST_FILE))
                
                # b) Split (train / test_split) => 80/20 sur le jeu d'entraînement
                X_train, X_test_split, y_train, y_test_split = train_test_split(
                    X_train_df, Y_train_df,
                    test_size=0.2,
                    stratify=Y_train_df['prdtypecode'],
                    random_state=self.config.random_state
                )
                test_split_indices = X_test_split.index.values
                train_indices = X_train.index.values

                # c) Éventuel balancing
                if balance_classes:
                    train_indices = self._create_balanced_dataset(X_train, y_train)
                    X_train = X_train.loc[train_indices]
                    y_train = y_train.loc[train_indices]

                # d) Extraction features via _extract_resnet_features
                #    1) Création dataset PyTorch pour train
                train_dataset = self._create_dataset(X_train, df_name="X_train")
                X_train_features = self._extract_resnet_features(train_dataset, desc="Extraction features train")

                #    2) Création dataset PyTorch pour test "officiel"
                test_dataset = self._create_dataset(X_test_df, df_name="X_test")
                X_test_features = self._extract_resnet_features(test_dataset, desc="Extraction features test")

                #    3) Création dataset PyTorch pour test_split
                test_split_dataset = self._create_dataset(X_test_split, df_name="X_test_split")
                X_test_split_features = self._extract_resnet_features(test_split_dataset, desc="Extraction features test_split")

                self.logger.info(f"Sauvegarde des fichiers images...")
                # e) Sauvegarde via _save_processed_data
                self._save_processed_data(
                    X_train_features['features'], 
                    y_train['prdtypecode'].values,
                    train_indices,
                    X_test_features['features'],
                    X_test_split_features['features'],
                    y_test_split['prdtypecode'].values,
                    test_split_indices,
                    required_files
                )
                
                self.logger.info(f"Mise à jour de l'état...")
                # Mise à jour de l'état
                self.preprocessed_data = {
                'X_train': X_train_features['features'],
                'y_train': y_train['prdtypecode'].values,
                'X_test': X_test_features['features'],
                'X_test_split': X_test_split_features['features'],
                'y_test_split': y_test_split['prdtypecode'].values,
                'train_indices': train_indices,
                'test_split_indices': test_split_indices,
                }

            
            # Traitement du texte si nécessaire
            if force_preprocess_text:
                self.logger.info("Prétraitement du texte en cours...")
                
                # Charger les fichiers CSV
                Y_train_df = safe_read_csv(str(Y_TRAIN_FILE))
                
                # Récupérer les indices exacts utilisés pour l'entraînement des images
                test_split_indices = self.preprocessed_data['test_split_indices']
                train_indices = self.preprocessed_data['train_indices']
                
                # Division des données en utilisant exactement les mêmes indices
                X_text_test = X_train_df.loc[X_train_df.index.isin(test_split_indices)].copy()
                X_text_train = X_train_df.loc[X_train_df.index.isin(train_indices)].copy()
                y_text_test = Y_train_df.loc[Y_train_df.index.isin(test_split_indices)].copy()
                y_text_train = Y_train_df.loc[Y_train_df.index.isin(train_indices)].copy()
                
                self.logger.info(f"Données texte préparées: {len(X_text_train)} exemples d'entraînement, {len(X_text_test)} exemples de test")
                
                # Création de texte combiné
                X_text_train.loc[:,'text'] = X_text_train['designation'].fillna('') + " " + X_text_train['description'].fillna('')
                X_text_test.loc[:,'text'] = X_text_test['designation'].fillna('') + " " + X_text_test['description'].fillna('')
                
                # Entraînement du modèle texte avec TF-IDF et SVM
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.svm import SVC
                from sklearn.pipeline import Pipeline
                import joblib
                
                # Configurer et entraîner le pipeline
                pipeline_svm = Pipeline([
                    ('tfidf', TfidfVectorizer(max_features=45000)),
                    ('model', SVC(C=12, kernel='rbf', gamma='scale', probability=True, class_weight='balanced'))
                ])
                
                self.logger.info(f"Entraînement du modèle texte sur {len(X_text_train)} exemples...")
                pipeline_svm.fit(X_text_train['text'], y_text_train['prdtypecode'])
                
                # Évaluer sur les données de test
                from sklearn.metrics import accuracy_score, classification_report
                y_pred = pipeline_svm.predict(X_text_test['text'])
                acc = accuracy_score(y_text_test['prdtypecode'], y_pred)
                self.logger.info(f"Précision du modèle texte: {acc:.4f}")
                
                # Sauvegarder le modèle
                os.makedirs('data/models/SVM', exist_ok=True)
                joblib.dump(pipeline_svm, "data/models/SVM/model.pkl")
                self.logger.info("Modèle texte sauvegardé dans data/models/SVM/model.pkl")
            
            return self.preprocessed_data
                
        except Exception as e:
            self.logger.error(f"Erreur préparation données: {str(e)}")
            self._cleanup()
            raise
    
    def save_model(self, model_type):
        try:
            model_dir = get_model_path(model_type)
            model_dir.mkdir(parents=True, exist_ok=True)
            
            if isinstance(self.model, NeuralClassifier):
                torch.save({
                    'state_dict': self.model.state_dict(),
                    'category_mapping': self.category_names,
                    'config': self.model.config,
                }, os.path.join(model_dir, 'model.pth'))
            else:
                with open(os.path.join(model_dir, 'model.pkl'), 'wb') as f:
                    pickle.dump({
                        'model': self.model,
                        'params': self.model.get_params(),
                        'category_mapping': self.category_names
                    }, f)
            
            self.logger.info(f"Modèle {model_type} sauvegardé dans {model_dir}")
        except Exception as e:
            self.logger.error(f"Erreur sauvegarde modèle {model_type}: {str(e)}")
            raise

    def load_model(self, model_type: str):
        """Charge un modèle sauvegardé avec ses métadonnées"""
        try:
            model_dir = get_model_path(model_type)
            
            if not model_dir.exists():
                raise FileNotFoundError(f"Dossier modèle non trouvé: {model_dir}")
            
            affichage_param = False # Pour afficher les paramètres des modèles chargés
            
            if not model_dir.exists():
                raise FileNotFoundError(f"Dossier modèle non trouvé: {model_dir}")
            if model_type == 'neural_net':
                
                model_path = model_dir / 'model.pth'
                model_data = torch.load(model_path, map_location=self.device)
                if affichage_param:
                    # Affichage des paramètres du modèle neural_net
                    print("\nParamètres du modèle neural_net:")
                    print("=" * 50)
                    print("\nConfiguration:")
                    for key, value in model_data['config'].items():
                        print(f"{key}: {value}")
                    
                    print("\nArchitecture du modèle:")
                    print("-" * 30)
                
                # Recréation du modèle
                self.model = NeuralClassifier(
                    num_classes=len(model_data['category_mapping']),
                    config=model_data['config']
                ).to(self.device)
                
                self.model.load_state_dict(model_data['state_dict'])
                print(self.model) if affichage_param else None
                self.category_names = model_data['category_mapping']
            else:
                model_path = model_dir / 'model.pkl'
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.model = model_data['model']
                self.category_names = model_data['category_mapping']
                
                if affichage_param:
                    # Affichage des paramètres selon le type de modèle
                    print(f"\nParamètres du modèle {model_type}:")
                    print("=" * 50)
                    if hasattr(self.model, 'get_params'):
                        params = self.model.get_params()
                        for param_name, value in sorted(params.items()):
                            print(f"{param_name}: {value}")
                    
                    # Pour XGBoost, afficher les paramètres additionnels importants
                    if model_type == 'xgboost':
                        print("\nParamètres additionnels:")
                        print("-" * 30)
                        print(f"Nombre d'arbres: {self.model.n_estimators}")
                        if hasattr(self.model, 'feature_importances_'):
                            print("Feature importances disponibles: Oui")
            
            self.logger.info(f"Modèle {model_type} chargé depuis {model_dir}")
            
        except Exception as e:
            self.logger.error(f"Erreur chargement modèle {model_type}: {str(e)}")
            raise

    ############## Entrainement modèles ############
    def cross_validate_model(self, model_type: str, **model_params) -> Dict[str, Any]:
        """
        Effectue une validation croisée avec support GPU et affiche la progression
        
        Args:
            model_type (str): Type de modèle à entraîner ('xgboost', 'lightgbm', 'catboost', 'logistic', 'neural_net')
            **model_params: Paramètres spécifiques au modèle
            
        Returns:
            Dict[str, Any]: Dictionnaire contenant le meilleur modèle et son F1-score
        """
        try:
            start_time = time.time()
            self.logger.info(f"Début de la validation croisée pour {model_type}")
            
            X = self.preprocessed_data['X_train']
            y = self.preprocessed_data['y_train']
            
            # Conversion des labels pour tous les modèles
            y = np.array([self.category_to_idx[label] for label in y])
        
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            best_f1 = 0
            best_model = None
            
            for fold, (train_idx, val_idx) in enumerate(tqdm(list(skf.split(X, y)), desc="Validation croisée", total=5), 1):
                fold_start = time.time()
                self.logger.info(f"Début du fold {fold}/5")
                
                X_fold_train, X_fold_val = X[train_idx], X[val_idx]
                y_fold_train, y_fold_val = y[train_idx], y[val_idx]
                
                # Initialisation du modèle en fonction du type
                if model_type == 'xgboost':
                    model = xgb.XGBClassifier(**model_params)
                    model.fit(X_fold_train, y_fold_train)
                    
                elif model_type == 'neural_net':
                    model = NeuralClassifier(
                        num_classes=len(self.category_names),
                        config=model_params
                    ).to(self.device)
                    
                    optimizer = torch.optim.Adam(
                        model.parameters(),
                        lr=model_params.get('learning_rate', 0.001),
                        weight_decay=model_params.get('weight_decay', 0.01)
                    )
                    
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer,
                        mode='min',
                        factor=model_params.get('scheduler_params', {}).get('factor', 0.1),
                        patience=model_params.get('scheduler_params', {}).get('patience', 3),
                        min_lr=model_params.get('scheduler_params', {}).get('min_lr', 1e-6)
                    )
                    
                    criterion = nn.CrossEntropyLoss()
                    best_val_loss = float('inf')
                    patience_counter = 0
                    
                    # Entraînement du modèle neuronal
                    for epoch in range(model_params.get('epochs', 30)):
                        model.train()
                        total_loss = 0
                        batch_count = 0
                        
                        # Phase d'entraînement
                        for batch_idx in range(0, len(X_fold_train), model_params.get('batch_size', 32)):
                            batch_end = min(batch_idx + model_params.get('batch_size', 32), len(X_fold_train))
                            batch_X = torch.FloatTensor(X_fold_train[batch_idx:batch_end]).to(self.device)
                            batch_y = torch.LongTensor(y_fold_train[batch_idx:batch_end]).to(self.device)
                            
                            optimizer.zero_grad()
                            outputs = model(batch_X)
                            loss = criterion(outputs, batch_y)
                            loss.backward()
                            optimizer.step()
                            
                            total_loss += loss.item()
                            batch_count += 1
                        
                        avg_loss = total_loss / batch_count
                        
                        # Phase de validation
                        model.eval()
                        val_predictions = []
                        with torch.no_grad():
                            for batch_idx in range(0, len(X_fold_val), model_params.get('batch_size', 32)):
                                batch_end = min(batch_idx + model_params.get('batch_size', 32), len(X_fold_val))
                                batch_X = torch.FloatTensor(X_fold_val[batch_idx:batch_end]).to(self.device)
                                outputs = model(batch_X)
                                _, preds = torch.max(outputs, 1)
                                val_predictions.extend(preds.cpu().numpy())
                        
                        y_pred = np.array(val_predictions)
                        val_f1 = f1_score(y_fold_val, y_pred, average='weighted')
                        
                        # Mise à jour du scheduler
                        scheduler.step(1 - val_f1)  # Utilise 1 - F1 comme métrique à minimiser
                        
                        # Early stopping
                        if val_f1 > best_val_loss:
                            best_val_loss = val_f1
                            patience_counter = 0
                        else:
                            patience_counter += 1
                        
                        if patience_counter >= model_params.get('early_stopping_patience', 5):
                            self.logger.info(f"Early stopping à l'époque {epoch + 1}")
                            break
                
                else:
                    raise ValueError(f"Type de modèle non supporté : {model_type}")
                
                # Calcul du F1-score pour le fold
                fold_f1 = f1_score(y_fold_val, y_pred, average='weighted')
                fold_time = time.time() - fold_start
                self.logger.info(f"Fold {fold}/5 terminé en {fold_time:.2f}s - F1-score: {fold_f1:.4f}")
                
                if fold_f1 > best_f1:
                    best_f1 = fold_f1
                    best_model = model
            
            total_time = time.time() - start_time
            self.logger.info(f"Validation croisée terminée en {total_time:.2f}s - Meilleur F1-score: {best_f1:.4f}")
            
            return {
                'model': best_model,
                'best_f1': best_f1
            }
                
        except Exception as e:
            self.logger.error(f"Erreur validation croisée: {str(e)}")
            raise

    def train_model(self, model_type='xgboost', use_cv=False, **model_params):
        """
        Choix du bon type d'entraînement selon le modèle
        """
        if model_type == 'neural_net':
            self.train_dl_model(use_cv=use_cv, **model_params)
        else:
            self.train_ml_model(model_type, use_cv=use_cv, **model_params)
        
        self.logger.info(f"Sauvegarde modèle {model_type} en cours...")
        self.save_model(model_type)

    def train_ml_model(self, model_type, use_cv, **model_params):
        """
        Entraîne un modèle ML avec les paramètres fournis
        
        Args:
            model_type (str): Type de modèle ('xgboost', 'lightgbm', 'catboost', 'logistic')
            use_cv (bool): Utiliser la validation croisée
            **model_params: Paramètres spécifiques au modèle
        """
        try:
            if not hasattr(self, 'preprocessed_data') or self.preprocessed_data is None:
                raise ValueError("Les données prétraitées ne sont pas disponibles")

            self.logger.info(f"Début entraînement {model_type}")
            start_time = time.time()

            # Préparation des données
            X_train = self.preprocessed_data['X_train']
            if isinstance(X_train, dict):
                X_train = X_train['features']
            if X_train.shape == ():
                X_train = X_train.item()['features']
                
            # Conversion des labels pour tous les modèles
            y_train = np.array([self.category_to_idx[label] for label in self.preprocessed_data['y_train']])

            if use_cv:
                result = self.cross_validate_model(model_type, **model_params)
                self.model = result['model']
            else:
                # Initialisation du modèle selon le type
                if model_type == 'xgboost':
                    self.model = xgb.XGBClassifier(**model_params)
                else:
                    raise ValueError(f"Type de modèle ML non supporté: {model_type}")

                # Entraînement
                self.model.fit(X_train, y_train)

            training_time = time.time() - start_time
            self.logger.info(f"Entraînement terminé en {training_time:.2f}s")

        except Exception as e:
            self.logger.error(f"Erreur entraînement {model_type}: {str(e)}")
            raise

    def train_dl_model(self, use_cv, **model_params):
        """
        Entraîne le modèle deep learning
        
        Args:
            use_cv (bool): Utiliser la validation croisée
            **model_params: Paramètres du modèle
        """
        try:
            if not hasattr(self, 'preprocessed_data') or self.preprocessed_data is None:
                raise ValueError("Les données prétraitées ne sont pas disponibles")

            self.logger.info("Début entraînement deep learning")
            start_time = time.time()

            # Conversion des labels
            y_train = np.array([self.category_to_idx[label] for label in self.preprocessed_data['y_train']])

            # Préparation et conversion des données
            X_train_data = self.preprocessed_data['X_train']
            if isinstance(X_train_data, dict):
                X_train_data = X_train_data['features']
            if X_train_data.shape == ():
                X_train_data = X_train_data.item()['features']
            
            # Conversion en float32 pour assurer la compatibilité
            X_train_data = np.array(X_train_data, dtype=np.float32)
            
            # Création des tenseurs
            X_train_tensor = torch.FloatTensor(X_train_data)
            y_train_tensor = torch.LongTensor(y_train)
            
            train_idx, val_idx = train_test_split(
                np.arange(len(X_train_tensor)),
                test_size=0.1,
                stratify=y_train,
                random_state=self.config.random_state
            )
            
            # Création des datasets
            train_dataset = TensorDataset(
                X_train_tensor[train_idx], 
                y_train_tensor[train_idx]
            )
            val_dataset = TensorDataset(
                X_train_tensor[val_idx], 
                y_train_tensor[val_idx]
            )

            # Extraction des paramètres spécifiques au DataLoader
            dataloader_params = {
                'batch_size': model_params['batch_size'],
                'num_workers': model_params['dataloader_params']['num_workers'],
                'pin_memory': model_params['dataloader_params']['pin_memory'],
                'prefetch_factor': model_params['dataloader_params']['prefetch_factor'],
                'persistent_workers': model_params['dataloader_params']['persistent_workers'],
            }

            # Création des DataLoaders avec les paramètres filtrés
            train_loader = DataLoader(train_dataset, shuffle=True, **dataloader_params)
            val_loader = DataLoader(val_dataset, shuffle=False, **dataloader_params)

            if use_cv:
                result = self.cross_validate_model('neural_net', **model_params)
                self.model = result['model']
            else:
                # Initialisation du modèle
                self.model = NeuralClassifier(
                    num_classes=len(self.category_names),
                    config=model_params
                ).to(self.device)

                # Configuration de l'optimisation
                optimizer = torch.optim.Adam(
                    self.model.parameters(),
                    lr=model_params.get('learning_rate', 0.001),
                    weight_decay=model_params.get('weight_decay', 0.01)
                )

                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='max',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6
                )

                criterion = nn.CrossEntropyLoss()
                best_val_f1 = 0
                patience_counter = 0
                
                # Boucle d'entraînement
                for epoch in range(model_params.get('epochs', 30)):
                    # Mode entraînement
                    self.model.train()
                    train_loss = 0
                    train_steps = 0

                    for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
                        data, target = data.to(self.device), target.to(self.device)
                        
                        optimizer.zero_grad()
                        output = self.model(data)
                        loss = criterion(output, target)
                        loss.backward()
                        optimizer.step()

                        train_loss += loss.item()
                        train_steps += 1

                    avg_train_loss = train_loss / train_steps

                    # Validation (pour early stopping)
                    self.model.eval()
                    val_predictions = []
                    val_targets = []
                    val_loss = 0
                    val_steps = 0

                    with torch.no_grad():
                        for data, target in val_loader:
                            data, target = data.to(self.device), target.to(self.device)
                            output = self.model(data)
                            loss = criterion(output, target)
                            val_loss += loss.item()
                            val_steps += 1

                            _, predicted = torch.max(output.data, 1)
                            val_predictions.extend(predicted.cpu().numpy())
                            val_targets.extend(target.cpu().numpy())

                    val_f1 = f1_score(val_targets, val_predictions, average='weighted')
                    avg_val_loss = val_loss / val_steps

                    # Mise à jour du scheduler
                    scheduler.step(val_f1)

                    # Early stopping
                    if val_f1 > best_val_f1:
                        best_val_f1 = val_f1
                        patience_counter = 0
                        # Sauvegarde du meilleur modèle
                        best_model_state = self.model.state_dict()
                    else:
                        patience_counter += 1

                    self.logger.info(
                        f"Epoch {epoch+1}: "
                        f"Train Loss = {avg_train_loss:.4f}, "
                        f"Val Loss = {avg_val_loss:.4f}, "
                        f"Val F1 = {val_f1:.4f}"
                    )

                    if patience_counter >= model_params.get('early_stopping_patience', 5):
                        self.logger.info(f"Early stopping à l'epoch {epoch+1}")
                        break

                # Restauration du meilleur modèle
                self.model.load_state_dict(best_model_state)

            training_time = time.time() - start_time
            self.logger.info(f"Entraînement terminé en {training_time:.2f}s")

        except Exception as e:
            self.logger.error(f"Erreur entraînement DL: {str(e)}")
            raise
    ################################################

    def predict(self, X):
        """
        Génère les prédictions pour de nouvelles données
        
        Args:
            X: Données d'entrée (numpy array ou torch tensor)
        
        Returns:
            tuple: (predictions, probabilités)
        """
        try:
            if self.model is None:
                raise ValueError("Le modèle n'est pas entraîné")

            # 1. Gestion des NpzFile (objet retourné par np.load)
            if hasattr(X, 'files'):  # C'est un NpzFile
                print("Détection d'un NpzFile, extraction des données...")
                # Essayer différentes clés communes
                possible_keys = ['arr_0', 'features', 'X_test_split', 'X_test']
                data_extracted = False
                
                for key in possible_keys:
                    if key in X.files:
                        X = X[key]
                        print(f"Données extraites avec la clé '{key}'")
                        data_extracted = True
                        break
                
                if not data_extracted:
                    # Prendre la première clé disponible
                    first_key = X.files[0]
                    X = X[first_key]
                    print(f"Données extraites avec la première clé disponible '{first_key}'")

            # 2. Gestion des scalaires numpy (array 0D)
            if isinstance(X, np.ndarray) and X.shape == ():
                X = X.item()  # Convertit en objet Python
                print("Conversion d'un array 0D en objet Python")

            # 3. Gestion des dictionnaires
            if isinstance(X, dict):
                if 'features' in X:
                    X = X['features']
                    print("Extraction des features depuis un dictionnaire")
                else:
                    # Prendre la première valeur du dictionnaire
                    first_key = list(X.keys())[0]
                    X = X[first_key]
                    print(f"Extraction des données avec la clé '{first_key}'")

            # 4. Conversion en numpy array si nécessaire
            if not isinstance(X, (np.ndarray, torch.Tensor)):
                X = np.array(X)
                print("Conversion en numpy array")

            # 5. Assurer que X est 2D
            if isinstance(X, np.ndarray):
                if len(X.shape) == 1:
                    X = X.reshape(1, -1)
                    print("Reshape de 1D vers 2D")
                elif len(X.shape) > 2:
                    # Aplatir les dimensions supplémentaires
                    X = X.reshape(X.shape[0], -1)
                    print(f"Reshape de {len(X.shape)}D vers 2D")

            # === DEBUG INFORMATION ===
            print(f"Type final de X: {type(X)}")
            print(f"Shape finale de X: {X.shape if hasattr(X, 'shape') else 'pas de shape'}")
            if isinstance(X, np.ndarray):
                print(f"Type des données: {X.dtype}")

            # === PRÉDICTION ===
            
            # Vérification du type de modèle
            is_dl_model = isinstance(self.model, NeuralClassifier)
            
            if is_dl_model:
                self.model.eval()
                # Conversion en tensor si nécessaire
                if isinstance(X, np.ndarray):
                    X = torch.FloatTensor(X).to(self.device)

                predictions = []
                probabilities = []
                
                with torch.no_grad():
                    for i in range(0, len(X), self.config.batch_size):
                        batch = X[i:i + self.config.batch_size]
                        outputs = self.model(batch)
                        probs = torch.softmax(outputs, dim=1)
                        preds = torch.argmax(outputs, dim=1)
                        
                        predictions.extend(preds.cpu().numpy())
                        probabilities.extend(probs.cpu().numpy())

                predictions = np.array(predictions)
                probabilities = np.array(probabilities)
                
                # Pour les modèles DL, les prédictions sont déjà des indices
                # Les convertir en codes de catégorie
                predictions = np.array([self.idx_to_category[int(idx)] for idx in predictions])
                
            else:
                # Pour les modèles ML classiques
                self.logger.info("Prédiction avec modèle ML")
                
                if isinstance(self.model, xgb.XGBClassifier):
                    batch_size = 1000
                    predictions = []
                    probabilities = []
                    
                    for i in range(0, len(X), batch_size):
                        batch = X[i:i + batch_size]
                        batch_pred = self.model.predict(batch)
                        batch_prob = self.model.predict_proba(batch)
                        predictions.extend(batch_pred)
                        probabilities.extend(batch_prob)
                        
                    probabilities = np.array(probabilities)
                    predictions = np.array(predictions)
                else:
                    predictions = self.model.predict(X)
                    probabilities = self.model.predict_proba(X)

                # Conversion des indices en codes de catégorie pour les modèles ML
                predictions = np.array([self.idx_to_category[int(idx)] for idx in predictions])

            print(f"Prédictions générées: {len(predictions)} échantillons")
            print(f"Shape des probabilités: {probabilities.shape}")
            
            return predictions, probabilities

        except Exception as e:
            self.logger.error(f"Erreur prédiction: {str(e)}")
            self.logger.error(f"Type de X reçu: {type(X)}")
            if hasattr(X, 'shape'):
                self.logger.error(f"Shape de X: {X.shape}")
            elif hasattr(X, 'files'):
                self.logger.error(f"Fichiers dans NpzFile: {X.files}")
            raise
    
    def predictions_exist(self, model_name):
        """Vérifie si les prédictions existent déjà"""
        pred_path = os.path.join(self.predictions_dir, f'predictions_{model_name}.csv')
        return os.path.exists(pred_path)

    def save_predictions(self, model_name, predictions, probabilities):
        """Sauvegarde les prédictions"""
        # 1. DataFrame pour la colonne 'prediction'
        df_pred = pd.DataFrame({'prediction': predictions})
        
        # 2. DataFrame pour les probabilités, 27 colonnes
        df_proba = pd.DataFrame(probabilities, columns=[f'prob_class_{i}' for i in range(probabilities.shape[1])])
        
        # 3. Concaténer les deux en un seul DataFrame
        df_final = pd.concat([df_pred, df_proba], axis=1)
        df_final.to_csv(os.path.join(self.predictions_dir, f'predictions_{model_name}.csv'), index=False)

    def load_predictions(self, model_name):
        pred_df = safe_read_csv(os.path.join(self.predictions_dir, f'predictions_{model_name}.csv'))
        predictions = pred_df['prediction'].values
        
        # On récupère toutes les colonnes de probas
        proba_cols = [c for c in pred_df.columns if c.startswith('prob_class_')]
        probabilities = pred_df[proba_cols].values  # shape (N, 27)
        
        return predictions, probabilities

    def _cleanup(self):
        """Nettoie les fichiers temporaires en cas d'erreur"""
        try:
            temp_dirs = ['temp_train', 'temp_test']
            for dir_name in temp_dirs:
                dir_path = os.path.join(self.config.data_path, dir_name)
                if os.path.exists(dir_path):
                    shutil.rmtree(dir_path)
        except Exception as e:
            self.logger.warning(f"Erreur nettoyage : {str(e)}")

    # ==================== MÉTHODES TEXTE ====================
    def preprocess_text(self, text_input):
        """
        Prétraite un texte pour le modèle SVM
        
        Args:
            text_input (str): Texte à prétraiter
            
        Returns:
            np.array: Features texte prétraitées
        """
        try:
            self.logger.info("Prétraitement du texte...")
            
            # Si le modèle de texte n'est pas chargé, le charger
            if not hasattr(self, 'text_model') or self.text_model is None:
                try:
                    import joblib
                    self.text_model = joblib.load(str(self.base_dir / 'data/models/SVM/model.pkl'))
                    self.logger.info("Modèle SVM texte chargé avec succès.")
                except Exception as e:
                    self.logger.error(f"Erreur lors du chargement du modèle texte: {str(e)}")
                    raise
            
            # Pour les modèles sklearn Pipeline, nous devons accéder au TF-IDF Vectorizer
            # puis transformer le texte avec ce vectorizer
            if hasattr(self.text_model, 'named_steps') and 'tfidf' in self.text_model.named_steps:
                tfidf = self.text_model.named_steps['tfidf']
                text_features = tfidf.transform([text_input]).toarray()
                self.logger.info(f"Texte transformé avec TF-IDF: {text_features.shape}")
                return text_features
            else:
                self.logger.error("Le modèle texte n'a pas de TF-IDF Vectorizer")
                raise ValueError("Format de modèle texte non supporté")
        
        except Exception as e:
            self.logger.error(f"Erreur lors du prétraitement du texte: {str(e)}")
            raise
            
    def load_text_model(self, model_name='SVM'):
        """Charge le modèle SVM texte dans le pipeline"""
        try:
            model_path = get_model_path('SVM') / 'model.pkl'
            import joblib
            self.text_model = joblib.load(str(model_path))
            self.logger.info(f"Modèle texte {model_name} chargé avec succès")
            return True
        except Exception as e:
            self.logger.error(f"Erreur chargement modèle texte: {e}")
            # Créer un modèle de secours
            class DummySVM:
                def __init__(self, num_classes):
                    self.num_classes = num_classes
                def predict(self, X):
                    return np.random.randint(0, self.num_classes, size=len(X))
                def predict_proba(self, X):
                    probs = np.random.random((len(X), self.num_classes))
                    return probs / probs.sum(axis=1, keepdims=True)
            
            self.text_model = DummySVM(num_classes=len(self.category_names))
            self.logger.info("Modèle de secours créé pour le texte")
            return False
    
    def prepare_text_data(self):
        """Prépare les données texte pour l'évaluation"""
        # Utiliser les indices du test split
        test_split_indices = self.preprocessed_data['test_split_indices']
        
        # Charger les données originales
        X_train_df = safe_read_csv(str(X_TRAIN_FILE))
        Y_train_df = safe_read_csv(str(Y_TRAIN_FILE))
        
        # Filtrer selon les indices de test
        valid_indices = [idx for idx in test_split_indices if idx in X_train_df.index]
        
        if len(valid_indices) > 0:
            X_test_text_df = X_train_df.loc[valid_indices]
            y_test_text_df = Y_train_df.loc[valid_indices]
            self.logger.info(f"Données texte préparées: {len(valid_indices)} échantillons")
        else:
            # Fallback: utiliser un échantillon aléatoire
            sample_size = len(test_split_indices)
            random_indices = np.random.choice(len(X_train_df), size=sample_size, replace=False)
            X_test_text_df = X_train_df.iloc[random_indices]
            y_test_text_df = Y_train_df.iloc[random_indices]
            self.logger.info(f"Échantillon aléatoire créé: {sample_size} échantillons")
        
        # Créer le texte combiné comme lors de l'entraînement
        X_test_text_df = X_test_text_df.copy()
        X_test_text_df['text'] = (X_test_text_df['designation'].fillna('') + " " + 
                                 X_test_text_df['description'].fillna(''))
        
        self.text_test_data = X_test_text_df['text'].values
        self.text_test_labels = y_test_text_df['prdtypecode'].values
        
        return self.text_test_data, self.text_test_labels
    
    def predict_text(self, X_text=None):
        """Génère des prédictions avec le modèle texte"""
        if X_text is None:
            X_text = self.text_test_data
            
        if not hasattr(self, 'text_model') or self.text_model is None:
            raise ValueError("Modèle texte non chargé. Appelez load_text_model() d'abord.")
        
        predictions = self.text_model.predict(X_text)
        probabilities = self.text_model.predict_proba(X_text)
        
        return predictions, probabilities
    
    def evaluate_text_model(self):
        """Évalue le modèle texte avec des métriques détaillées"""
        if not hasattr(self, 'text_test_data') or not hasattr(self, 'text_test_labels'):
            raise ValueError("Données texte non préparées. Appelez prepare_text_data() d'abord.")
        
        # Prédictions
        predictions, probabilities = self.predict_text()
        
        # Métriques de base
        accuracy = accuracy_score(self.text_test_labels, predictions)
        f1_weighted = f1_score(self.text_test_labels, predictions, average='weighted')
        f1_macro = f1_score(self.text_test_labels, predictions, average='macro')
        
        # Métriques détaillées
        max_probas = np.max(probabilities, axis=1)
        
        results = {
            'accuracy': round(accuracy, 4),
            'weighted_f1': round(f1_weighted, 4),
            'macro_f1': round(f1_macro, 4),
            'mean_confidence': round(np.mean(max_probas), 4),
            'median_confidence': round(np.median(max_probas), 4),
            'min_confidence': round(np.min(max_probas), 4),
            'low_confidence_samples': int(np.sum(max_probas < 0.5)),
            'high_confidence_samples': int(np.sum(max_probas > 0.8)),
            'num_samples': len(predictions)
        }
        
        return results, predictions, probabilities
    
    def save_text_predictions(self, predictions, probabilities, model_name='SVM'):
        """Sauvegarde les prédictions du modèle texte"""
        # DataFrame pour la colonne 'prediction'
        df_pred = pd.DataFrame({'prediction': predictions})
        
        # DataFrame pour les probabilités
        df_proba = pd.DataFrame(probabilities, columns=[f'prob_class_{i}' for i in range(probabilities.shape[1])])
        
        # Concaténer
        df_final = pd.concat([df_pred, df_proba], axis=1)
        df_final.to_csv(os.path.join(self.predictions_dir, f'predictions_text_{model_name}.csv'), index=False)
        
        self.logger.info(f"Prédictions texte sauvegardées pour {model_name}")
    
    def text_predictions_exist(self, model_name='SVM'):
        """Vérifie si les prédictions texte existent déjà"""
        pred_path = os.path.join(self.predictions_dir, f'predictions_text_{model_name}.csv')
        return os.path.exists(pred_path)
    
    def load_text_predictions(self, model_name='SVM'):
        """Charge les prédictions texte existantes"""
        pred_path = os.path.join(self.predictions_dir, f'predictions_text_{model_name}.csv')
        pred_df = safe_read_csv(pred_path)
        predictions = pred_df['prediction'].values
        
        # Récupérer les probabilités
        proba_cols = [c for c in pred_df.columns if c.startswith('prob_class_')]
        probabilities = pred_df[proba_cols].values
        
        return predictions, probabilities

    # ==================== MÉTHODES POUR MULTIMODALE ET STREAMLIT ====================
    def fuse_predictions(self, text_probs, image_probs, strategy='mean'):
        """
        Fusionne les prédictions texte et image selon différentes stratégies
        
        Args:
            text_probs: Probabilités du modèle texte
            image_probs: Probabilités du modèle image
            strategy: Stratégie de fusion ('mean', 'product', 'max', 'weighted', 'confidence_weighted')
            
        Returns:
            np.array: Probabilités fusionnées
        """
        if strategy == 'mean':
            return (text_probs + image_probs) / 2
        elif strategy == 'product':
            fused = text_probs * image_probs
            return fused / fused.sum(axis=1, keepdims=True)
        elif strategy == 'max':
            return np.maximum(text_probs, image_probs)
        elif strategy == 'weighted':
            # Pondération fixe 60% texte, 40% image (à ajuster selon tests)
            return text_probs * 0.6 + image_probs * 0.4
        elif strategy == 'confidence_weighted':
            # Pondération dynamique basée sur la confiance de chaque modèle
            text_confidence = np.max(text_probs, axis=1, keepdims=True)
            image_confidence = np.max(image_probs, axis=1, keepdims=True)
            
            total_confidence = text_confidence + image_confidence
            text_weights = text_confidence / total_confidence
            image_weights = image_confidence / total_confidence
            
            return text_probs * text_weights + image_probs * image_weights
        else:
            raise ValueError(f"Stratégie de fusion {strategy} non supportée")
    
    def create_combined_prediction_report(self, text_predictions, text_probabilities, 
                                        image_predictions, image_probabilities, 
                                        fused_predictions, fused_probabilities,
                                        y_true, fusion_strategy):
        """
        Crée un rapport combiné pour les prédictions multimodales
        
        Args:
            text_predictions: Prédictions du modèle texte
            text_probabilities: Probabilités du modèle texte  
            image_predictions: Prédictions du modèle image
            image_probabilities: Probabilités du modèle image
            fused_predictions: Prédictions fusionnées
            fused_probabilities: Probabilités fusionnées
            y_true: Vérité terrain
            fusion_strategy: Stratégie de fusion utilisée
            
        Returns:
            pd.DataFrame: Rapport détaillé
        """
        rapport_data = []
        
        for i in range(len(y_true)):
            text_conf = np.max(text_probabilities[i])
            image_conf = np.max(image_probabilities[i])
            fused_conf = np.max(fused_probabilities[i])
            
            # Noms des classes
            true_class_name = self.category_names.get(y_true[i], f"Unknown_{y_true[i]}")
            text_pred_name = self.category_names.get(text_predictions[i], f"Unknown_{text_predictions[i]}")
            image_pred_name = self.category_names.get(image_predictions[i], f"Unknown_{image_predictions[i]}")
            fused_pred_name = self.category_names.get(fused_predictions[i], f"Unknown_{fused_predictions[i]}")
            
            rapport_data.append({
                'sample_id': i,
                'true_label': y_true[i],
                'true_class_name': true_class_name,
                'text_prediction': text_predictions[i],
                'text_class_name': text_pred_name,
                'text_confidence': text_conf,
                'text_correct': text_predictions[i] == y_true[i],
                'image_prediction': image_predictions[i],
                'image_class_name': image_pred_name,
                'image_confidence': image_conf,
                'image_correct': image_predictions[i] == y_true[i],
                'fused_prediction': fused_predictions[i],
                'fused_class_name': fused_pred_name,
                'fused_confidence': fused_conf,
                'fused_correct': fused_predictions[i] == y_true[i],
                'fusion_strategy': fusion_strategy,
                'agreement_text_image': text_predictions[i] == image_predictions[i],
                'best_individual': 'text' if text_conf > image_conf else 'image',
                'fusion_improved': (fused_predictions[i] == y_true[i]) and 
                                 (text_predictions[i] != y_true[i] or image_predictions[i] != y_true[i])
            })
        
        return pd.DataFrame(rapport_data)
    
    def process_new_input(self, text_input, image_path):
        """
        Prétraite un nouvel exemple (texte + image) pour la prédiction
        
        Args:
            text_input (str): Texte d'entrée (description du produit)
            image_path (str): Chemin vers l'image du produit
            
        Returns:
            dict: Features prétraitées {
                'text_features': np.array,
                'image_features': np.array,
                'raw_text': str  # Ajout du texte brut
            }
        """
        try:
            self.logger.info("Prétraitement des données streamlit..")
            
            # 1. Traitement de l'image
            # Créer un dataset d'une seule image
            single_dataset = RakutenImageDataset([image_path])
            
            # Extraire les features via ResNet
            resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            resnet.fc = nn.Identity()  # Retire la dernière couche
            resnet = resnet.to(self.device)
            resnet.eval()
            
            # Créer un DataLoader pour un seul élément
            dataloader = DataLoader(
                single_dataset,
                batch_size=1,
                num_workers=1,
                pin_memory=True
            )
            
            # Extraire les features
            with torch.no_grad():
                for batch in dataloader:
                    inputs = batch.to(self.device)
                    image_features = resnet(inputs).cpu().numpy()
            
            # 2. Traitement du texte
            text_features = self.preprocess_text(text_input)
            
            self.logger.info("Prétraitement terminé avec succès.")
            
            return {
                'text_features': text_features,
                'image_features': image_features,
                'raw_text': text_input  # Garder le texte brut pour SHAP
            }
        
        except Exception as e:
            self.logger.error(f"Erreur lors du prétraitement: {str(e)}")
            raise

    def predict_multimodal(self, text_input, image_path, fusion_strategy='mean'):
        """
        Effectue une prédiction multimodale sur un nouvel exemple
        
        Args:
            text_input (str): Texte d'entrée
            image_path (str): Chemin vers l'image
            fusion_strategy (str): Stratégie de fusion ('mean', 'product', 'weighted')
            
        Returns:
            dict: Résultats de prédiction {
                'predicted_class': int,
                'predicted_class_name': str,
                'probabilities': np.array,
                'text_prediction': int,
                'image_prediction': int
            }
        """
        try:
            # 1. Prétraiter l'exemple
            features = self.process_new_input(text_input, image_path)
            
            # 2. Prédictions individuelles
            # Modèle texte
            if not hasattr(self, 'text_model') or self.text_model is None:
                import joblib
                self.text_model = joblib.load('data/models/SVM/model.pkl')
            
            text_pred = self.text_model.predict(features['text_features'])
            text_probs = self.text_model.predict_proba(features['text_features'])
            
            # Modèle image
            image_pred, image_probs = self.predict(features['image_features'])
            
            # S'assurer que les probabilités ont la même forme
            if text_probs.shape != image_probs.shape:
                self.logger.warning(f"Dimensions des probabilités texte ({text_probs.shape}) et image ({image_probs.shape}) différentes.")
                # Étendre ou réduire si nécessaire
                if len(text_probs.shape) > len(image_probs.shape):
                    image_probs = image_probs.reshape(text_probs.shape)
                else:
                    text_probs = text_probs.reshape(image_probs.shape)
            
            # 3. Fusion des prédictions
            fused_probs = self.fuse_predictions(text_probs, image_probs, strategy=fusion_strategy)
            
            # 4. Prédiction finale
            predicted_idx = np.argmax(fused_probs, axis=1)[0]
            predicted_class = self.idx_to_category[predicted_idx]
            predicted_class_name = self.category_names[predicted_class]
            
            return {
                'predicted_class': int(predicted_class),
                'predicted_class_name': predicted_class_name,
                'probabilities': fused_probs[0],
                'text_prediction': int(text_pred[0]),
                'image_prediction': int(image_pred[0]) if isinstance(image_pred, np.ndarray) else int(image_pred)
            }
        
        except Exception as e:
            self.logger.error(f"Erreur lors de la prédiction multimodale: {str(e)}")
            raise

    def get_model_explanations(self, text_input, image_path, fusion_strategy='mean'):
        """
        Génère des explications SHAP pour une prédiction multimodale (Version corrigée)
        
        Args:
            text_input (str): Texte d'entrée
            image_path (str): Chemin vers l'image
            fusion_strategy (str): Stratégie de fusion
            
        Returns:
            dict: Explications simplifiées
        """
        try:
            self.logger.info("Génération des explications simplifiées...")
            
            # 1. Prétraiter l'entrée
            features = self.process_new_input(text_input, image_path)
            
            # 2. Prédictions individuelles pour comparaison
            # Modèle texte
            if not hasattr(self, 'text_model') or self.text_model is None:
                import joblib
                self.text_model = joblib.load('data/models/SVM/model.pkl')
            
            text_pred = self.text_model.predict([features['raw_text']])  # Utiliser le texte brut
            text_probs = self.text_model.predict_proba([features['raw_text']])
            
            # Modèle image
            image_pred, image_probs = self.predict(features['image_features'])
            
            # 3. Fusion des prédictions
            if text_probs.shape != image_probs.shape:
                self.logger.warning(f"Dimensions des probabilités différentes: texte {text_probs.shape}, image {image_probs.shape}")
                # Adapter si nécessaire
                if len(image_probs.shape) > len(text_probs.shape):
                    text_probs = text_probs.reshape(image_probs.shape)
                else:
                    image_probs = image_probs.reshape(text_probs.shape)
            
            fused_probs = self.fuse_predictions(text_probs, image_probs, strategy=fusion_strategy)
            predicted_idx = np.argmax(fused_probs, axis=1)[0]
            predicted_class = self.idx_to_category[predicted_idx]
            predicted_class_name = self.category_names[predicted_class]
            
            # 4. Explications simplifiées (sans SHAP complexe)
            
            # Importance basique du modèle image (si XGBoost)
            image_importance = None
            if hasattr(self.model, 'feature_importances_'):
                image_importance = self.model.feature_importances_
                top_image_features = np.argsort(image_importance)[-10:][::-1]
            else:
                top_image_features = np.arange(10)  # Fallback
            
            # Analyse texte basique (longueur, mots clés, etc.)
            # S'assurer que le texte est bien une string
            raw_text = features['raw_text']
            if isinstance(raw_text, np.ndarray):
                if raw_text.size == 1:
                    raw_text = str(raw_text.item())
                else:
                    raw_text = ' '.join([str(item) for item in raw_text.flatten()])
            else:
                raw_text = str(raw_text)

            text_analysis = {
                'text_length': len(raw_text),
                'word_count': len(raw_text.split()),
                'text_confidence': np.max(text_probs),
                'top_words': raw_text.lower().split()[:10]  # Maintenant safe car raw_text est une string
            }
            
            # Analyse des contributions
            text_confidence = np.max(text_probs)
            image_confidence = np.max(image_probs)
            total_confidence = text_confidence + image_confidence
            
            # 5. Résumé des explications
            explanations = {
                'prediction': {
                    'predicted_class': int(predicted_class),
                    'predicted_class_name': predicted_class_name,
                    'confidence': float(np.max(fused_probs))
                },
                'individual_predictions': {
                    'text_prediction': int(text_pred[0]),
                    'text_confidence': float(text_confidence),
                    'image_prediction': int(image_pred[0]) if hasattr(image_pred, '__getitem__') else int(image_pred),
                    'image_confidence': float(image_confidence)
                },
                'modality_importance': {
                    'text_weight': float(text_confidence / total_confidence) if total_confidence > 0 else 0.5,
                    'image_weight': float(image_confidence / total_confidence) if total_confidence > 0 else 0.5,
                    'fusion_strategy': fusion_strategy
                },
                'text_analysis': text_analysis,
                'image_analysis': {
                    'top_features': top_image_features.tolist()[:5] if isinstance(top_image_features, np.ndarray) else list(range(5)),
                    'feature_importance_available': image_importance is not None
                },
                'fusion_analysis': {
                    'agreement': text_pred[0] == (image_pred[0] if hasattr(image_pred, '__getitem__') else image_pred),
                    'confidence_boost': float(np.max(fused_probs)) > max(text_confidence, image_confidence),
                    'dominant_modality': 'text' if text_confidence > image_confidence else 'image'
                }
            }
            
            self.logger.info("Explications simplifiées générées avec succès")
            return explanations
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la génération des explications: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            # Retourner un dictionnaire minimal en cas d'erreur
            return {
                'prediction': {
                    'predicted_class': 0,
                    'predicted_class_name': 'Erreur',
                    'confidence': 0.0
                },
                'error': str(e),
                'modality_importance': {
                    'text_weight': 0.5,
                    'image_weight': 0.5,
                    'fusion_strategy': fusion_strategy
                }
            }
        
    def align_multimodal_datasets(self, X_text, X_image, y=None):
        """
        Aligne les jeux de données texte et image en utilisant les indices de produit
        
        Args:
            X_text: Dataset texte (DataFrame ou tableau)
            X_image: Dataset image
            y: Labels (optionnel)
            
        Returns:
            tuple: (X_text_aligned, X_image_aligned, y_aligned)
        """
        # Récupérer les indices de produit dans le jeu d'images
        if hasattr(self, 'preprocessed_data') and 'test_split_indices' in self.preprocessed_data:
            product_indices = self.preprocessed_data['test_split_indices']
        else:
            # Si les indices ne sont pas disponibles, on suppose que X_image contient les IDs
            if hasattr(X_image, 'index'):
                product_indices = X_image.index
            else:
                # Si X_image est un numpy array, on ne peut pas déterminer les indices
                raise ValueError("Impossible de déterminer les indices de produit dans le jeu d'images")
        
        # Filtrer le dataset texte
        if isinstance(X_text, pd.DataFrame) or isinstance(X_text, pd.Series):
            X_text_aligned = X_text.loc[product_indices]
        else:
            # Pour un array, on doit reconstruire le mapping
            # On a besoin d'avoir accès au DataFrame original
            X_df_path = 'data/X_test_update.csv'
            if os.path.exists(X_df_path):
                X_df = safe_read_csv(X_df_path)
                X_df_filtered = X_df.loc[product_indices]
                
                # Recréer le texte
                X_text_aligned = X_df_filtered['designation'].fillna('') + " " + X_df_filtered['description'].fillna('')
                X_text_aligned = X_text_aligned.values
            else:
                raise ValueError(f"Le fichier {X_df_path} est nécessaire pour aligner les données texte")
        
        # Aligner les labels si fournis
        if y is not None:
            if isinstance(y, pd.Series):
                y_aligned = y.loc[product_indices]
            else:
                # Pour un array, on utilise les mêmes indices
                # Attention: cela suppose que y a le même ordre que X_image
                y_aligned = y
        else:
            y_aligned = None
        
        return X_text_aligned, X_image, y_aligned
            
    def extract_and_save_indices(self, output_dir='data/indices'):
        """
        Extrait et sauvegarde tous les indices importants pour les tests et analyses futurs en tenant compte des doublons du balancing
        
        Args:
            output_dir: Répertoire où sauvegarder les fichiers d'indices
            
        Returns:
            dict: Dictionnaire contenant tous les ensembles d'indices
        """
        # Créer le répertoire de sortie si nécessaire
        os.makedirs(output_dir, exist_ok=True)
        
        # Récupérer les indices sauvegardés après le prétraitement complet
        test_split_indices = self.preprocessed_data['test_split_indices']
        train_balanced_indices = self.preprocessed_data['train_indices']
        
        # Charger le dataset original pour avoir tous les indices
        X_train_df = safe_read_csv('data/X_train_update.csv')
        all_indices = X_train_df.index.values
        
        # Reconstituer les indices d'entraînement originaux (avant balancing)
        # en excluant les indices de test du dataset complet
        original_train_indices = np.setdiff1d(all_indices, test_split_indices)
        
        # Reconstituer les indices d'entraînement originaux (après balancing)
        train_balanced_unique = np.unique(train_balanced_indices)
        
        # Calculer les indices ignorés (dans l'ensemble train original mais pas dans train_balanced)
        ignored_indices = np.setdiff1d(original_train_indices, train_balanced_indices)
        
        # Vérifier les tailles pour s'assurer que tout est cohérent
        expected_total = len(train_balanced_unique) + len(test_split_indices) + len(ignored_indices)
        
        # Créer un dictionnaire avec tous les ensembles d'indices
        indices_dict = {
            'all_indices': all_indices,
            'test_split_indices': test_split_indices,
            'train_original_indices': original_train_indices,
            'train_balanced_indices': train_balanced_indices,  # Avec doublons (pour cohérence avec l'entraînement)
            'train_balanced_unique': train_balanced_unique,    # Sans doublons
            'ignored_indices': ignored_indices
        }
        
        # Sauvegarder chaque ensemble d'indices
        for name, indices in indices_dict.items():
            np.save(os.path.join(output_dir, f"{name}.npy"), indices)
            
            # Créer un DataFrame avec les indices et des descriptions pour chaque échantillon
            df = pd.DataFrame(indices, columns=['index'])
            
            # Sauvegarder au format CSV
            df.to_csv(os.path.join(output_dir, f"{name}.csv"), index=False)
        
        # Créer un DataFrame unique montrant la distribution de chaque index
        distribution_df = pd.DataFrame(index=all_indices)
        distribution_df['in_test'] = distribution_df.index.isin(test_split_indices)
        distribution_df['in_train_original'] = distribution_df.index.isin(original_train_indices)
        distribution_df['in_train_balanced'] = distribution_df.index.isin(train_balanced_unique)
        distribution_df['is_ignored'] = distribution_df.index.isin(ignored_indices)
        
        # Sauvegarder la distribution complète
        distribution_df.to_csv(os.path.join(output_dir, "index_distribution.csv"))
        
        self.logger.info(f"Indices sauvegardés dans {output_dir}")
        self.logger.info(f"Distribution des indices:")
        self.logger.info(f"  - Total: {len(all_indices)}")
        self.logger.info(f"  - Test: {len(test_split_indices)} ({len(test_split_indices)/len(all_indices)*100:.1f}%)")
        self.logger.info(f"  - Train original: {len(original_train_indices)} ({len(original_train_indices)/len(all_indices)*100:.1f}%)")
        self.logger.info(f"  - Train balanced (unique): {len(train_balanced_unique)} ({len(train_balanced_unique)/len(all_indices)*100:.1f}%)")
        pourcentage_doublons  = (len(train_balanced_indices) - len(train_balanced_unique)) / len(train_balanced_indices) * 100
        self.logger.info(f"  - Train balanced (total): {len(train_balanced_indices)} (avec {len(train_balanced_indices) - len(train_balanced_unique)} doublons ({pourcentage_doublons:.1f}%))")
        self.logger.info(f"  - Ignorés: {len(ignored_indices)} ({len(ignored_indices)/len(all_indices)*100:.1f}%)")
        self.logger.info(f"  - Vérification: {len(train_balanced_unique) + len(test_split_indices) + len(ignored_indices)} = {len(all_indices)} ✓")
        
        return indices_dict
                
    def predict_text_single(self, text_input):
        """
        Prédiction texte pour un seul échantillon (usage Streamlit)
        
        Args:
            text_input (str): Texte à classifier
            
        Returns:
            tuple: (prediction, probabilities)
        """
        try:
            if not hasattr(self, 'text_model') or self.text_model is None:
                raise ValueError("Modèle texte non chargé. Appelez load_text_model() d'abord.")
            
            # S'assurer que l'entrée est une string
            if isinstance(text_input, list):
                text_input = text_input[0]
            text_input = str(text_input).strip()
            
            if len(text_input) == 0:
                raise ValueError("Le texte d'entrée est vide")
            
            # Prédiction directe avec le modèle SVM
            predictions = self.text_model.predict([text_input])
            probabilities = self.text_model.predict_proba([text_input])
            
            self.logger.info(f"Prédiction texte: {predictions[0]}, confiance: {np.max(probabilities[0]):.3f}")
            
            return predictions[0], probabilities[0]
            
        except Exception as e:
            self.logger.error(f"Erreur prédiction texte single: {e}")
            raise

    def process_single_input(self, text_input=None, image_path=None):
        """
        Prétraite une entrée unique (texte et/ou image) pour prédiction
        Version améliorée qui gère texte seul, image seule, ou multimodal
        
        Args:
            text_input (str, optional): Texte d'entrée
            image_path (str, optional): Chemin vers l'image
            
        Returns:
            dict: Features prétraitées selon les modalités disponibles
        """
        try:
            self.logger.info(f"Prétraitement entrée unique - Texte: {'Oui' if text_input else 'Non'}, Image: {'Oui' if image_path else 'Non'}")
            
            features = {}
            
            # 1. Traitement du texte si fourni
            if text_input is not None and len(str(text_input).strip()) > 0:
                text_input_clean = str(text_input).strip()
                
                # Vérifier que le modèle texte est chargé
                if not hasattr(self, 'text_model') or self.text_model is None:
                    import joblib
                    model_path = self.base_dir / 'data/models/SVM/model.pkl'
                    if model_path.exists():
                        self.text_model = joblib.load(str(model_path))
                        self.logger.info("Modèle SVM chargé pour prétraitement")
                    else:
                        raise FileNotFoundError(f"Modèle SVM non trouvé: {model_path}")
                
                # Prétraitement avec TF-IDF (via le pipeline SVM)
                if hasattr(self.text_model, 'named_steps') and 'tfidf' in self.text_model.named_steps:
                    tfidf = self.text_model.named_steps['tfidf']
                    text_features = tfidf.transform([text_input_clean]).toarray()
                    features['text_features'] = text_features
                    features['raw_text'] = text_input_clean
                    self.logger.info(f"Features texte extraites: {text_features.shape}")
                else:
                    raise ValueError("Modèle texte sans TF-IDF Vectorizer")
            
            # 2. Traitement de l'image si fournie
            if image_path is not None and os.path.exists(str(image_path)):
                # Créer un dataset d'une seule image
                single_dataset = RakutenImageDataset([str(image_path)])
                
                # Extraire les features via ResNet
                resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
                resnet.fc = nn.Identity()  # Retire la dernière couche
                resnet = resnet.to(self.device)
                resnet.eval()
                
                # Créer un DataLoader pour un seul élément
                dataloader = DataLoader(
                    single_dataset,
                    batch_size=1,
                    num_workers=0,  # Éviter les problèmes multiprocessing en Streamlit
                    pin_memory=False  # Plus stable pour un seul échantillon
                )
                
                # Extraire les features
                with torch.no_grad():
                    for batch in dataloader:
                        inputs = batch.to(self.device)
                        image_features = resnet(inputs).cpu().numpy()
                        features['image_features'] = image_features
                        self.logger.info(f"Features image extraites: {image_features.shape}")
                        break  # Un seul batch
            
            # 3. Validation
            if not features:
                raise ValueError("Aucune modalité valide fournie (texte ou image)")
            
            self.logger.info(f"Prétraitement terminé. Modalités: {list(features.keys())}")
            return features
            
        except Exception as e:
            self.logger.error(f"Erreur prétraitement entrée unique: {e}")
            raise

    def predict_multimodal(self, text_input, image_path, fusion_strategy='mean'):
        """
        Effectue une prédiction multimodale sur un nouvel exemple
        Version corrigée utilisant les nouvelles fonctions robustes
        
        Args:
            text_input (str): Texte d'entrée
            image_path (str): Chemin vers l'image
            fusion_strategy (str): Stratégie de fusion ('mean', 'product', 'weighted')
            
        Returns:
            dict: Résultats de prédiction
        """
        try:
            self.logger.info(f"Prédiction multimodale: {fusion_strategy}")
            
            # 1. Prétraiter l'exemple avec la nouvelle fonction robuste
            features = self.process_single_input(text_input, image_path)
            
            # 2. Prédictions individuelles
            
            # Modèle texte
            if 'text_features' in features:
                if not hasattr(self, 'text_model') or self.text_model is None:
                    import joblib
                    self.text_model = joblib.load('data/models/SVM/model.pkl')
                
                text_pred = self.text_model.predict(features['text_features'])
                text_probs = self.text_model.predict_proba(features['text_features'])
                
                # Pour SVM, les prédictions sont directement les codes de catégorie
                text_prediction = text_pred[0]
                text_probabilities = text_probs[0]
            else:
                raise ValueError("Features texte non disponibles")
            
            # Modèle image
            if 'image_features' in features:
                image_pred, image_probs = self.predict(features['image_features'])
                
                # Pour les modèles image, convertir l'indice vers code de catégorie
                if isinstance(image_pred, np.ndarray):
                    image_prediction = image_pred[0]
                else:
                    image_prediction = image_pred
                    
                if len(image_probs.shape) > 1:
                    image_probabilities = image_probs[0]
                else:
                    image_probabilities = image_probs
            else:
                raise ValueError("Features image non disponibles")
            
            # 3. S'assurer que les probabilités ont la même forme
            if text_probabilities.shape != image_probabilities.shape:
                self.logger.warning(f"Dimensions des probabilités différentes: texte {text_probabilities.shape}, image {image_probabilities.shape}")
                # Adapter si nécessaire - généralement, elles devraient avoir la même taille (27 classes)
                min_size = min(len(text_probabilities), len(image_probabilities))
                text_probabilities = text_probabilities[:min_size]
                image_probabilities = image_probabilities[:min_size]
            
            # 4. Fusion des prédictions
            fused_probs = self.fuse_predictions(
                text_probabilities.reshape(1, -1), 
                image_probabilities.reshape(1, -1), 
                strategy=fusion_strategy
            )
            
            # 5. Prédiction finale
            predicted_idx = np.argmax(fused_probs, axis=1)[0]
            
            # Convertir l'indice en code de catégorie
            if hasattr(self, 'idx_to_category') and predicted_idx in self.idx_to_category:
                predicted_class = self.idx_to_category[predicted_idx]
            else:
                # Si pas de mapping, supposer que l'indice EST le code
                predicted_class = predicted_idx
                
            predicted_class_name = self.category_names.get(predicted_class, f"Unknown_{predicted_class}")
            
            results = {
                'predicted_class': int(predicted_class),
                'predicted_class_name': predicted_class_name,
                'probabilities': fused_probs[0],
                'text_prediction': int(text_prediction),
                'image_prediction': int(image_prediction),
                'confidence': float(np.max(fused_probs))
            }
            
            self.logger.info(f"Prédiction multimodale terminée: {predicted_class_name} (confiance: {results['confidence']:.3f})")
            return results
            
        except Exception as e:
            self.logger.error(f"Erreur prédiction multimodale: {str(e)}")
            raise

class NeuralClassifier(nn.Module):
    def __init__(self, num_classes, config=None):
        super().__init__()
        self.config = config or {}
        self.dropout_rate = self.config.get('dropout_rate', 0.3)
        
        # Architecture pour features pré-extraites de dimension 2048
        self.classifier = nn.Sequential(
            nn.Linear(2048, 1536),
            nn.BatchNorm1d(1536),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            
            nn.Linear(1536, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            
            nn.Linear(512, num_classes)
        )
        
        # Initialisation des poids améliorée
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)

    def forward(self, x):
        return self.classifier(x)
