#!/usr/bin/env python3
import sys
import requests
import zipfile
import os

def download_adownload_processed_filesnd_extract_zip(url, output_zip="Preprocessed_data.zip", extract_dir="."):
    """
    Télécharge un fichier ZIP depuis `url` et l'extrait dans `extract_dir`.
    """
    # 1) Téléchargement
    print(f"Téléchargement du ZIP depuis {url} ...")
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        print(f"Erreur : statut HTTP {response.status_code}")
        sys.exit(1)

    # Écriture binaire du contenu dans un fichier .zip local
    with open(output_zip, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print(f"Téléchargement terminé. ZIP enregistré sous : {output_zip}")

    # 2) Extraction
    print(f"Extraction du contenu de l'archive dans : {extract_dir}")
    # Crée le dossier d'extraction s'il n'existe pas déjà
    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(output_zip, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    print("Extraction terminée !")