#!/usr/bin/env python3
import sys
import zipfile
import os
import gdown
import shutil

def telecharger_et_extraire_zip(url, fichier_zip="Preprocessed_data.zip", dossier_extraction="."):
    """
    Télécharge un fichier ZIP depuis Google Drive et l'extrait.
    """
    try:
        # 1) Téléchargement avec gdown
        print(f"Téléchargement du fichier depuis Google Drive...")
        gdown.download(url, fichier_zip, fuzzy=True)
        
        # 2) Vérification et extraction
        if not os.path.exists(fichier_zip):
            raise FileNotFoundError("Le téléchargement a échoué")
            
        if not zipfile.is_zipfile(fichier_zip):
            raise zipfile.BadZipFile("Le fichier téléchargé n'est pas un ZIP valide")
            

        # 3) Création d'un dossier temporaire pour l'extraction
        temp_dir = "temp_extraction"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)
        
        # 4) Extraction dans le dossier temporaire
        print(f"Extraction temporaire...")
        with zipfile.ZipFile(fichier_zip, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
            
        # 5) Déplacement des fichiers vers le dossier final
        print(f"Déplacement des fichiers vers : {dossier_extraction}")
        os.makedirs(dossier_extraction, exist_ok=True)
        
        # Déplacer le contenu du sous-dossier processed_data vers le dossier final
        source_dir = os.path.join(temp_dir, "processed_data")
        for item in os.listdir(source_dir):
            s = os.path.join(source_dir, item)
            d = os.path.join(dossier_extraction, item)
            if os.path.exists(d):
                if os.path.isdir(d):
                    shutil.rmtree(d)
                else:
                    os.remove(d)
            shutil.move(s, d)
        
        print("Extraction terminée avec succès !")
        
    except Exception as e:
        print(f"Erreur : {e}")
        sys.exit(1)
        
    finally:
        # Nettoyage dans tous les cas
        print("Nettoyage des fichiers temporaires...")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        if os.path.exists(fichier_zip):
            os.remove(fichier_zip)
