import pandas as pd
import numpy as np
from preprocess import ProductClassificationPipeline, PipelineConfig
from utils import safe_read_csv
import os
import logging

def regenerer_rapport_modele(model_type, pipeline, nouveaux_libelles):
    """
    Regénère un rapport pour un modèle donné
    
    Args:
        model_type (str): Type de modèle ('neural_net', 'xgboost', 'SVM')
        pipeline: Instance du pipeline avec les données chargées
        nouveaux_libelles (dict): Dictionnaire des nouveaux libellés
    
    Returns:
        pd.DataFrame: Rapport généré
    """
    print(f"🔄 Régénération du rapport {model_type}...")
    
    try:
        # 1) Charger le modèle spécifique
        print(f"🧠 Chargement du modèle {model_type}...")
        if model_type == 'SVM':
            # Pour SVM, utiliser le modèle texte  
            pipeline.load_text_model('SVM')
            pipeline.prepare_text_data()
            print("✅ Modèle SVM (texte) chargé")
        else:
            # Pour neural_net et xgboost, utiliser les modèles image
            pipeline.load_model(model_type)
            print(f"✅ Modèle {model_type} chargé")
        
        # ⚠️ FORCER les nouveaux libellés après chargement (écrase les anciens sauvegardés dans le modèle)
        anciens_libelles = pipeline.category_names.copy() if hasattr(pipeline, 'category_names') else {}
        pipeline.category_names = nouveaux_libelles
        print(f"🔄 Libellés forcés vers les nouveaux pour {model_type}")
        
        # Afficher quelques comparaisons anciennes vs nouvelles
        if anciens_libelles and model_type != 'SVM':  # Pour les modèles image seulement
            print("   📋 Comparaison libellés (quelques exemples):")
            for code in list(nouveaux_libelles.keys())[:3]:
                ancien = anciens_libelles.get(code, 'Non défini')
                nouveau = nouveaux_libelles.get(code, 'Non défini')
                if ancien != nouveau:
                    print(f"      Code {code}: '{ancien}' → '{nouveau}'")
        
        # 2) Préparer les données selon le type de modèle
        if model_type == 'SVM':
            # Données texte
            X_test = pipeline.text_test_data
            y_test = pipeline.text_test_labels
            test_indices = pipeline.preprocessed_data['test_split_indices']
            
            # Prédictions texte
            predictions, probabilities = pipeline.predict_text(X_test)
            
        else:
            # Données image
            X_test_split = pipeline.preprocessed_data['X_test_split']
            y_test_split = pipeline.preprocessed_data['y_test_split']
            test_indices = pipeline.preprocessed_data['test_split_indices']
            
            # Gestion du format des données
            if isinstance(X_test_split, dict):
                X_test_split = X_test_split['features']
            elif X_test_split.shape == ():
                X_test_split = X_test_split.item()['features']
                
            # Prédictions image
            predictions, probabilities = pipeline.predict(X_test_split)
            y_test = y_test_split
        
        print(f"✅ Prédictions {model_type} générées pour {len(predictions)} échantillons")
        
        # 3) Créer le DataFrame du rapport
        print("📊 Création du rapport...")
        
        # Charger les données originales pour récupérer imageid
        X_train_df = safe_read_csv('../data/X_train_update.csv')
        
        rapport_data = []
        for i, test_idx in enumerate(test_indices):
            if test_idx in X_train_df.index:
                image_info = X_train_df.loc[test_idx]
                
                # Récupération des données
                predicted_category = predictions[i]
                true_category = y_test[i]
                prediction_probability = np.max(probabilities[i])
                
                # Génération des noms de catégories avec NOUVEAUX libellés
                predicted_category_name = nouveaux_libelles.get(predicted_category, f"Catégorie {predicted_category}")
                true_category_name = nouveaux_libelles.get(true_category, f"Catégorie {true_category}")
                
                # Vérification que les nouveaux libellés sont utilisés
                if i == 0:  # Première itération seulement
                    print(f"   🔍 Vérification libellés - Code {predicted_category} → '{predicted_category_name}'")
                
                rapport_data.append({
                    'original_index': i,
                    'imageid': image_info['imageid'],
                    'predicted_category': predicted_category,
                    'prediction_probability': round(prediction_probability, 8),
                    'true_category': true_category,
                    'predicted_category_name': predicted_category_name,
                    'true_category_name': true_category_name
                })
        
        # Créer le DataFrame
        rapport_df = pd.DataFrame(rapport_data)
        
        # 4) Sauvegarder le nouveau fichier
        if model_type == 'SVM':
            output_file = '../data/reports/rapport_text_SVM_nouveau.csv'
        else:
            output_file = f'../data/reports/rapport_{model_type.lower()}_nouveau.csv'
            
        rapport_df.to_csv(output_file, index=False)
        
        print(f"✅ Rapport sauvegardé dans {output_file}")
        print(f"📈 Statistiques du rapport {model_type}:")
        print(f"   - Nombre d'échantillons: {len(rapport_df)}")
        print(f"   - Codes de catégories utilisés: {sorted(rapport_df['predicted_category'].unique())}")
        print(f"   - Précision: {(rapport_df['predicted_category'] == rapport_df['true_category']).mean():.4f}")
        
        # 5) Vérifier que tous les codes ont des libellés
        codes_sans_libelle = []
        for code in rapport_df['predicted_category'].unique():
            if nouveaux_libelles.get(code, '').startswith('Catégorie ') or nouveaux_libelles.get(code, '') == '':
                codes_sans_libelle.append(code)
                
        if codes_sans_libelle:
            print(f"\n⚠️ ATTENTION: Codes sans libellé défini: {codes_sans_libelle}")
        else:
            print(f"\n✅ Tous les codes {model_type} ont des libellés définis")
            
        return rapport_df
        
    except Exception as e:
        print(f"❌ Erreur lors de la régénération {model_type}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def regenerer_tous_les_rapports():
    """
    Regénère tous les rapports (SVM, neural_net, xgboost)
    """
    print("🚀 Régénération de tous les rapports...")
    
    try:
        # 1) Configuration et initialisation du pipeline
        config = PipelineConfig(
            data_path='../data',
            model_path='../data/models', 
            image_dir='../data/images',
            batch_size=128,
            target_size=2000,
            num_workers=7
        )
        
        pipeline = ProductClassificationPipeline(config)
        print("✅ Pipeline initialisé")
        
        # 2) Charger les données préprocessées
        print("📁 Chargement des données préprocessées...")
        pipeline.preprocessed_data = pipeline.prepare_data_streamlit()
        print("✅ Données chargées")
        
        # 3) FORCER les nouveaux libellés (avant de charger les modèles)
        nouveaux_libelles = {
            1: "Livres occasion",
            4: "Jeux consoles neuf", 
            5: "Accessoires gaming",
            6: "Consoles de jeux",
            13: "Modélisme",
            114: "Objets pop culture",
            116: "Cartes de jeux",
            118: "Jeux de rôle et figurines",
            128: "Jouets enfant",
            132: "Puériculture",
            156: "Mobilier",
            192: "Linge de maison",
            194: "Épicerie",
            206: "Décoration",
            222: "Animalerie",
            228: "Journaux et revues occasion",
            1281: "Jeux enfant", 
            1301: "Lingerie enfant et jeu de bar",
            1302: "Jeux et accessoires de plein air",
            2403: "Lots livres et magazines",
            2462: "Console et Jeux vidéos occasion",
            2522: "Fournitures papeterie",
            2582: "Mobilier et accessoires de jardin",
            2583: "Piscine et accessoires",
            2585: "Outillage de jardin",
            2705: "Livres neufs",
            2905: "Jeux PC en téléchargement"
        }
        
        # 3) Liste des modèles à traiter
        modeles = ['SVM', 'neural_net', 'xgboost']
        rapports = {}
        
        # 4) Regénérer chaque rapport
        for model_type in modeles:
            print(f"\n{'='*50}")
            print(f"📊 TRAITEMENT DU MODÈLE {model_type.upper()}")
            print(f"{'='*50}")
            
            try:
                rapport = regenerer_rapport_modele(model_type, pipeline, nouveaux_libelles)
                rapports[model_type] = rapport
                
                # Afficher quelques exemples
                print(f"\n📋 Exemples du rapport {model_type}:")
                print(rapport.head(5).to_string(index=False))
                
            except Exception as e:
                print(f"❌ Erreur avec le modèle {model_type}: {str(e)}")
                rapports[model_type] = None
        
        # 5) Sauvegarder les indices utilisés (une seule fois)
        sauvegarder_indices_utilises(pipeline)
        
        # 6) Résumé final
        print(f"\n{'='*50}")
        print("📊 RÉSUMÉ FINAL")
        print(f"{'='*50}")
        
        for model_type, rapport in rapports.items():
            if rapport is not None:
                precision = (rapport['predicted_category'] == rapport['true_category']).mean()
                print(f"✅ {model_type}: {len(rapport)} échantillons, précision: {precision:.4f}")
            else:
                print(f"❌ {model_type}: Échec de la génération")
        
        print(f"\n📁 FICHIERS GÉNÉRÉS:")
        print("📋 Rapports principaux:")
        for model_type in modeles:
            prefix = "text_" if model_type == 'SVM' else ""
            print(f"   ✅ ../data/reports/rapport_{prefix}{model_type.lower()}_nouveau.csv")
            
        print("\n📊 Analyses d'erreurs:")
        for model_type in modeles:
            print(f"   ✅ ../data/erreurs/analyse_erreurs_{model_type.lower()}.json")
            print(f"   ✅ ../data/erreurs/erreurs_par_classe_{model_type.lower()}.csv")
            print(f"   ✅ ../data/erreurs/matrice_confusion_{model_type.lower()}.csv")
            
        print("\n💾 Prédictions:")
        for model_type in modeles:
            prefix = "text_" if model_type == 'SVM' else ""
            print(f"   ✅ ../data/predictions/predictions_{prefix}{model_type.lower()}.csv")
            
        print("\n🔍 Explications:")
        for model_type in modeles:
            print(f"   ✅ ../data/explanations/explications_{model_type.lower()}.json")
            
        print("\n📋 Indices:")
        print("   ✅ ../data/indices/indices_utilises.json")
        print("   ✅ ../data/indices/test_indices.csv")
        
        return rapports
        
    except Exception as e:
        print(f"❌ Erreur lors de la régénération complète: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def analyser_erreurs_predictions(model_type, pipeline, nouveaux_libelles, predictions, probabilities, y_true, test_indices):
    """
    Génère les analyses d'erreurs détaillées pour un modèle
    
    Args:
        model_type (str): Type de modèle 
        pipeline: Instance du pipeline
        nouveaux_libelles (dict): Nouveaux libellés des catégories
        predictions: Prédictions du modèle
        probabilities: Probabilités des prédictions
        y_true: Vraies étiquettes
        test_indices: Indices de test
        
    Returns:
        dict: Analyses d'erreurs générées
    """
    print(f"📊 Analyse des erreurs pour {model_type}...")
    
    try:
        from sklearn.metrics import confusion_matrix, classification_report
        import json
        
        # 1) Matrice de confusion
        cm = confusion_matrix(y_true, predictions)
        
        # 2) Rapport de classification détaillé
        class_report = classification_report(y_true, predictions, output_dict=True, zero_division=0)
        
        # 3) Analyse par classe
        analyses_par_classe = {}
        for class_code in nouveaux_libelles.keys():
            if class_code in y_true or class_code in predictions:
                # Indices des échantillons de cette classe
                true_indices = np.where(y_true == class_code)[0]
                pred_indices = np.where(predictions == class_code)[0]
                
                if len(true_indices) > 0:
                    # Prédictions correctes/incorrectes pour cette classe
                    correct_mask = (y_true[true_indices] == predictions[true_indices])
                    
                    # Confiances pour cette classe
                    class_probs = probabilities[true_indices]
                    if len(class_probs.shape) > 1:
                        # Trouver l'indice de la classe dans les probabilités
                        if hasattr(pipeline, 'category_to_idx'):
                            class_prob_idx = pipeline.category_to_idx.get(class_code, 0)
                        else:
                            class_prob_idx = 0
                        confidence_scores = class_probs[:, class_prob_idx] if class_probs.shape[1] > class_prob_idx else class_probs[:, 0]
                    else:
                        confidence_scores = class_probs
                    
                    analyses_par_classe[class_code] = {
                        'nom_classe': nouveaux_libelles[class_code],
                        'nb_echantillons': len(true_indices),
                        'nb_correct': int(np.sum(correct_mask)),
                        'nb_incorrect': int(len(true_indices) - np.sum(correct_mask)),
                        'precision': float(correct_mask.mean()) if len(correct_mask) > 0 else 0.0,
                        'confiance_moyenne': float(np.mean(confidence_scores)) if len(confidence_scores) > 0 else 0.0,
                        'confiance_min': float(np.min(confidence_scores)) if len(confidence_scores) > 0 else 0.0,
                        'confiance_max': float(np.max(confidence_scores)) if len(confidence_scores) > 0 else 0.0,
                        'echantillons_faible_confiance': int(np.sum(confidence_scores < 0.5)) if len(confidence_scores) > 0 else 0
                    }
        
        # 4) Analyse globale des erreurs
        correct_predictions = (y_true == predictions)
        max_probs = np.max(probabilities, axis=1)
        
        analyse_globale = {
            'accuracy': float(np.mean(correct_predictions)),
            'nb_total_echantillons': len(y_true),
            'nb_correct': int(np.sum(correct_predictions)),
            'nb_incorrect': int(np.sum(~correct_predictions)),
            'confiance_moyenne_globale': float(np.mean(max_probs)),
            'confiance_moyenne_correct': float(np.mean(max_probs[correct_predictions])) if np.sum(correct_predictions) > 0 else 0.0,
            'confiance_moyenne_incorrect': float(np.mean(max_probs[~correct_predictions])) if np.sum(~correct_predictions) > 0 else 0.0,
            'nb_predictions_faible_confiance': int(np.sum(max_probs < 0.5))
        }
        
        # 5) Sauvegarder les analyses
        error_analysis = {
            'model_type': model_type,
            'analyse_globale': analyse_globale,
            'analyses_par_classe': analyses_par_classe,
            'matrice_confusion': cm.tolist(),
            'rapport_classification': class_report,
            'nouveaux_libelles': nouveaux_libelles
        }
        
        # Sauvegarder dans le dossier erreurs
        os.makedirs('../data/erreurs', exist_ok=True)
        
        # Fichier JSON principal
        with open(f'../data/erreurs/analyse_erreurs_{model_type.lower()}.json', 'w', encoding='utf-8') as f:
            json.dump(error_analysis, f, indent=2, ensure_ascii=False)
        
        # Fichier CSV détaillé par classe
        classes_df = pd.DataFrame.from_dict(analyses_par_classe, orient='index')
        classes_df.index.name = 'code_classe'
        classes_df.to_csv(f'../data/erreurs/erreurs_par_classe_{model_type.lower()}.csv')
        
        # Matrice de confusion en CSV
        cm_df = pd.DataFrame(cm)
        cm_df.to_csv(f'../data/erreurs/matrice_confusion_{model_type.lower()}.csv')
        
        print(f"✅ Analyses d'erreurs {model_type} sauvegardées")
        return error_analysis
        
    except Exception as e:
        print(f"❌ Erreur analyse erreurs {model_type}: {str(e)}")
        return {}

def sauvegarder_predictions_detaillees(model_type, predictions, probabilities, test_indices):
    """
    Sauvegarde les prédictions dans le format attendu par Streamlit
    """
    print(f"💾 Sauvegarde prédictions {model_type}...")
    
    try:
        os.makedirs('../data/predictions', exist_ok=True)
        
        # DataFrame principal
        pred_data = {
            'prediction': predictions,
            'test_index': test_indices[:len(predictions)]  # S'assurer que les tailles correspondent
        }
        
        # Ajouter les probabilités
        for i in range(probabilities.shape[1]):
            pred_data[f'prob_class_{i}'] = probabilities[:, i]
        
        pred_df = pd.DataFrame(pred_data)
        
        # Sauvegarder
        if model_type == 'SVM':
            filename = f'../data/predictions/predictions_text_{model_type.lower()}.csv'
        else:
            filename = f'../data/predictions/predictions_{model_type.lower()}.csv'
            
        pred_df.to_csv(filename, index=False)
        print(f"✅ Prédictions {model_type} sauvegardées dans {filename}")
        
    except Exception as e:
        print(f"❌ Erreur sauvegarde prédictions {model_type}: {str(e)}")

def generer_explications_modele(model_type, pipeline, nouveaux_libelles):
    """
    Génère les fichiers d'explication pour l'interprétabilité
    """
    print(f"🔍 Génération explications {model_type}...")
    
    try:
        os.makedirs('../data/explanations', exist_ok=True)
        
        explications = {
            'model_type': model_type,
            'categories': nouveaux_libelles,
            'nb_classes': len(nouveaux_libelles),
            'architecture': 'Neural Network' if model_type == 'neural_net' else model_type.upper()
        }
        
        # Feature importance si disponible
        if hasattr(pipeline.model, 'feature_importances_'):
            explications['feature_importances'] = pipeline.model.feature_importances_.tolist()
            
        # Infos sur le modèle
        if hasattr(pipeline.model, 'get_params'):
            explications['parametres_modele'] = pipeline.model.get_params()
            
        # Sauvegarder
        import json
        with open(f'../data/explanations/explications_{model_type.lower()}.json', 'w', encoding='utf-8') as f:
            json.dump(explications, f, indent=2, ensure_ascii=False)
            
        print(f"✅ Explications {model_type} générées")
        
    except Exception as e:
        print(f"❌ Erreur explications {model_type}: {str(e)}")

def sauvegarder_indices_utilises(pipeline):
    """
    Sauvegarde les indices utilisés pour la reproductibilité
    """
    print("📋 Sauvegarde des indices utilisés...")
    
    try:
        os.makedirs('../data/indices', exist_ok=True)
        
        indices_info = {
            'test_split_indices': pipeline.preprocessed_data['test_split_indices'].tolist(),
            'nb_test_samples': len(pipeline.preprocessed_data['test_split_indices']),
            'train_indices': pipeline.preprocessed_data['train_indices'].tolist() if 'train_indices' in pipeline.preprocessed_data else [],
            'nb_train_samples': len(pipeline.preprocessed_data['train_indices']) if 'train_indices' in pipeline.preprocessed_data else 0
        }
        
        # Sauvegarder en JSON et CSV
        import json
        with open('../data/indices/indices_utilises.json', 'w') as f:
            json.dump(indices_info, f, indent=2)
            
        # CSV pour les indices de test
        test_df = pd.DataFrame({'test_index': pipeline.preprocessed_data['test_split_indices']})
        test_df.to_csv('../data/indices/test_indices.csv', index=False)
        
        print("✅ Indices sauvegardés")
        
    except Exception as e:
        print(f"❌ Erreur sauvegarde indices: {str(e)}")
    """
    Compare les anciens et nouveaux rapports pour tous les modèles
    """
    try:
        print("\n🔍 Comparaison avec les anciens rapports...")
        
        for model_type in modeles:
            print(f"\n📊 COMPARAISON {model_type.upper()}:")
            print("-" * 40)
            
            # Noms des fichiers
            if model_type == 'SVM':
                ancien_file = '../data/reports/rapport_text_SVM.csv'
                nouveau_file = '../data/reports/rapport_text_SVM_nouveau.csv'
            else:
                ancien_file = f'../data/reports/rapport_{model_type.lower()}.csv'
                nouveau_file = f'../data/reports/rapport_{model_type.lower()}_nouveau.csv'
            
            # Charger l'ancien rapport si il existe
            if os.path.exists(ancien_file):
                ancien_df = safe_read_csv(ancien_file)
                print(f"   📂 Ancien: {len(ancien_df)} échantillons")
                
                # Afficher les codes uniques dans l'ancien rapport
                if 'predicted_category' in ancien_df.columns:
                    anciens_codes = ancien_df['predicted_category'].unique()
                    print(f"   📋 Anciens codes: {sorted(anciens_codes)[:10]}...")
                
                if os.path.exists(nouveau_file):
                    nouveau_df = safe_read_csv(nouveau_file)
                    nouveaux_codes = nouveau_df['predicted_category'].unique()
                    print(f"   📋 Nouveaux codes: {sorted(nouveaux_codes)[:10]}...")
                    
                    # Comparaison des libellés
                    print("   🔄 Exemples de changements:")
                    for i in range(min(3, len(ancien_df), len(nouveau_df))):
                        ancien_libelle = ancien_df.iloc[i].get('predicted_category_name', 'N/A')
                        nouveau_libelle = nouveau_df.iloc[i]['predicted_category_name']
                        if ancien_libelle != nouveau_libelle:
                            print(f"      Ligne {i}: '{ancien_libelle}' → '{nouveau_libelle}'")
                else:
                    print(f"   ⚠️ Nouveau fichier {nouveau_file} non trouvé")
            else:
                print(f"   📂 Ancien fichier {ancien_file} non trouvé")
            
    except Exception as e:
        print(f"⚠️ Erreur lors de la comparaison: {str(e)}")

def nettoyer_et_remplacer_fichiers(confirmer=False):
    """
    Remplace les anciens fichiers par les nouveaux (optionnel)
    """
    if not confirmer:
        print("\n❓ Pour remplacer les anciens fichiers, exécutez:")
        print("   nettoyer_et_remplacer_fichiers(confirmer=True)")
        return
        
    print("\n🔄 Remplacement des anciens fichiers...")
    
    remplacements = [
        ('../data/reports/rapport_text_SVM_nouveau.csv', '../data/reports/rapport_text_SVM.csv'),
        ('../data/reports/rapport_neural_net_nouveau.csv', '../data/reports/rapport_neural_net.csv'),
        ('../data/reports/rapport_xgboost_nouveau.csv', '../data/reports/rapport_xgboost.csv')
    ]
    
    for nouveau, ancien in remplacements:
        if os.path.exists(nouveau):
            try:
                # Sauvegarder l'ancien comme backup
                if os.path.exists(ancien):
                    backup = ancien.replace('.csv', '_backup.csv')
                    os.rename(ancien, backup)
                    print(f"   📦 Backup créé: {backup}")
                
                # Remplacer par le nouveau
                os.rename(nouveau, ancien)
                print(f"   ✅ {ancien} remplacé")
                
            except Exception as e:
                print(f"   ❌ Erreur remplacement {ancien}: {e}")
        else:
            print(f"   ⚠️ {nouveau} non trouvé")

if __name__ == "__main__":
    # Configuration logging pour réduire le bruit
    logging.getLogger('classification_pipeline').setLevel(logging.WARNING)
    
    # Regénérer tous les rapports
    rapports = regenerer_tous_les_rapports()
    
    # Comparer avec les anciens
    comparer_anciens_nouveaux_rapports()
    
    print("\n🎯 Régénération terminée!")
    print("   - Nouveaux fichiers créés avec le suffixe '_nouveau'")
    print("   - Vérifiez les résultats avant de remplacer les anciens")
    print("   - Pour remplacer: nettoyer_et_remplacer_fichiers(confirmer=True)")
    
    # Afficher les fichiers créés
    nouveaux_fichiers = [
        '../data/reports/rapport_text_SVM_nouveau.csv',
        '../data/reports/rapport_neural_net_nouveau.csv', 
        '../data/reports/rapport_xgboost_nouveau.csv'
    ]
    
    print(f"\n📁 Fichiers générés:")
    for fichier in nouveaux_fichiers:
        if os.path.exists(fichier):
            print(f"   ✅ {fichier}")
        else:
            print(f"   ❌ {fichier} (non créé)")