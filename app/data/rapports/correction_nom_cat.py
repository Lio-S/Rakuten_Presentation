import pandas as pd

def corriger_fichiers_csv():
    """
    Remplace simplement les anciens noms de catégories par les nouveaux
    """
    
    # Mapping des codes vers les nouveaux noms
    nouveaux_noms_par_code = {
        10: "Livres occasion",
        40: "Jeux consoles neuf", 
        50: "Accessoires gaming",
        60: "Consoles de jeux",
        1140: "Objets pop culture",
        1160: "Cartes de jeux",
        1180: "Jeux de rôle et figurines",
        1280: "Jouets enfant",
        1281: "Jeux enfant", 
        1300: "Modélisme",
        1301: "Chaussettes enfant",
        1302: "Jeux de plein air",
        1320: "Puériculture",
        1560: "Mobilier",
        1920: "Linge de maison",
        1940: "Épicerie",
        2060: "Décoration",
        2220: "Animalerie",
        2280: "Journaux et revues occasion",
        2403: "Lots livres et magazines",
        2462: "Jeux vidéos occasion",
        2522: "Fournitures papeterie",
        2582: "Mobilier de jardin",
        2583: "Piscine et accessoires",
        2585: "Outillage de jardin",
        2705: "Livres neufs",
        2905: "Jeux PC"
    }
    
    # Mapping des anciens noms vers les nouveaux noms (pour remplacement textuel)
    remplacement_noms = {
        "Livres": "Livres occasion",
        "DVD & Films": "Jeux consoles neuf",
        "Jouets & Jeux": "Accessoires gaming", 
        "Consoles": "Consoles de jeux",
        "TV": "Objets pop culture",
        "Électroménager": "Cartes de jeux",
        "Décoration": "Jeux de rôle et figurines",
        "Accessoires téléphones": "Jouets enfant",
        "Téléphonie fixe": "Jeux enfant",
        "Jeux vidéo ancien": "Modélisme",
        "Consoles rétro": "Chaussettes enfant",
        "Consoles rÃ©tro": "Chaussettes enfant",  # Version avec caractères encodés
        "Jeux vidéo rétro": "Jeux de plein air",
        "CD": "Puériculture",
        "Photos": "Mobilier",
        "Musique amplifiée": "Linge de maison",
        "Instrument musique": "Épicerie",
        "Articles soins": "Décoration",
        "Puériculture": "Animalerie",
        "Jeux vidéo": "Journaux et revues occasion",
        "Livres en langues étrangères": "Lots livres et magazines",
        "Fournitures bureau": "Jeux vidéos occasion",
        "Équipement bébé": "Fournitures papeterie",
        "Matériel & accessoires": "Mobilier de jardin",
        "Articles sport": "Piscine et accessoires",
        "Sports & Loisirs": "Outillage de jardin",
        "Accessoires console": "Livres neufs",
        "Instruments musique": "Jeux PC"
    }
    
    # Fonction pour remplacer les noms (remplacement textuel)
    def remplacer_nom(nom):
        if pd.isna(nom):
            return nom
        return remplacement_noms.get(nom, nom)
    
    # Fonction pour obtenir le nom basé sur le code
    def obtenir_nom_par_code(code):
        return nouveaux_noms_par_code.get(code, f"Catégorie {code}")
    
    # Traitement du fichier SVM
    print("Correction du fichier rapport_text_SVM.csv...")
    try:
        df_svm = pd.read_csv('rapport_text_SVM.csv')
        
        # Remplacer dans toutes les colonnes contenant des noms de catégories
        for col in df_svm.columns:
            if 'class_name' in col or 'category_name' in col:
                df_svm[col] = df_svm[col].apply(remplacer_nom)
        
        df_svm.to_csv('rapport_text_SVM_corrige.csv', index=False)
        print("✓ rapport_text_SVM_corrige.csv créé")
        
    except Exception as e:
        print(f"Erreur avec rapport_text_SVM.csv: {e}")
    
    # Traitement du fichier Neural Net (cas spécial car les noms ne correspondent pas aux codes)
    print("\nCorrection du fichier rapport_neural_net.csv...")
    try:
        df_neural = pd.read_csv('rapport_neural_net.csv')
        
        # Corriger predicted_category_name basé sur predicted_category
        if 'predicted_category' in df_neural.columns and 'predicted_category_name' in df_neural.columns:
            def obtenir_nom_par_code(code):
                return nouveaux_noms_par_code.get(code, f"Catégorie {code}")
            
            df_neural['predicted_category_name'] = df_neural['predicted_category'].apply(obtenir_nom_par_code)
        
        # Corriger true_category_name basé sur true_category si elle existe
        if 'true_category' in df_neural.columns:
            if 'true_category_name' not in df_neural.columns:
                df_neural['true_category_name'] = ""
            df_neural['true_category_name'] = df_neural['true_category'].apply(lambda code: nouveaux_noms_par_code.get(code, f"Catégorie {code}"))
        
        # Remplacer dans les autres colonnes contenant des noms de catégories (au cas où)
        for col in df_neural.columns:
            if ('class_name' in col or 'category_name' in col) and col not in ['predicted_category_name', 'true_category_name']:
                df_neural[col] = df_neural[col].apply(remplacer_nom)
        
        df_neural.to_csv('rapport_neural_net_corrige.csv', index=False)
        print("✓ rapport_neural_net_corrige.csv créé")
        
    except Exception as e:
        print(f"Erreur avec rapport_neural_net.csv: {e}")
    
    # Traitement du fichier XGBoost (même logique que neural_net au cas où)
    print("\nCorrection du fichier rapport_xgboost.csv...")
    try:
        df_xgb = pd.read_csv('rapport_xgboost.csv')
        
        # Corriger predicted_category_name basé sur predicted_category si les colonnes existent
        if 'predicted_category' in df_xgb.columns and 'predicted_category_name' in df_xgb.columns:
            df_xgb['predicted_category_name'] = df_xgb['predicted_category'].apply(obtenir_nom_par_code)
        
        # Corriger true_category_name basé sur true_category si elle existe
        if 'true_category' in df_xgb.columns:
            if 'true_category_name' not in df_xgb.columns:
                df_xgb['true_category_name'] = ""
            df_xgb['true_category_name'] = df_xgb['true_category'].apply(obtenir_nom_par_code)
        
        # Remplacer dans les autres colonnes contenant des noms de catégories
        for col in df_xgb.columns:
            if ('class_name' in col or 'category_name' in col) and col not in ['predicted_category_name', 'true_category_name']:
                df_xgb[col] = df_xgb[col].apply(remplacer_nom)
        
        df_xgb.to_csv('rapport_xgboost_corrige.csv', index=False)
        print("✓ rapport_xgboost_corrige.csv créé")
        
    except Exception as e:
        print(f"Erreur avec rapport_xgboost.csv: {e}")
    
    print("\n=== Correction terminée ===")
    print("- SVM: Remplacement textuel des anciens noms")
    print("- Neural Net: Correction basée sur les codes (predicted_category, true_category)")
    print("- XGBoost: Correction basée sur les codes + remplacement textuel")
    print("\nLes fichiers corrigés ont été créés avec le suffixe '_corrige'")
    print("\nExemples de corrections Neural Net:")
    print("  Code 10 -> 'Livres occasion'")
    print("  Code 24 -> 'Catégorie 24' (code non mappé)")
    print("  Code 1301 -> 'Chaussettes enfant'")


corriger_fichiers_csv()