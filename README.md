# Projet Rakuten - Installation

## Prérequis

- **Python 3.10.12** (obligatoire)
- Environnement virtuel (recommandé)
- Streamlit >= 1.28.0

## Installation locale
```bash
pyenv install 3.10.12
pyenv local 3.10.12
pip install -r requirements.txt
streamlit run app.py
```

Pour tout problème d'installation, vérifier :
1. Version Python : `python --version`
2. Pip à jour : `pip --version`
3. Espace disque suffisant : ~2GB requis