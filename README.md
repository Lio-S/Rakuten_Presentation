# Projet Rakuten - Installation

## Prérequis

- **Python 3.10.12** (obligatoire)
- Environnement virtuel (recommandé)

## Installation automatique

### Linux / WSL2 / macOS
```bash
chmod +x install.sh
./install.sh
```

### Windows
```cmd
install.bat
```

## Installation manuelle

### 1. Préparer l'environnement
```bash
# Créer un environnement virtuel
python -m venv venv

# Activer l'environnement
# Linux/macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

### 2. Installer les dépendances
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Vérification de l'installation

```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

## Utilisation avec Jupyter Lab

### Démarrer Jupyter Lab
```bash
jupyter lab
```

### Sélectionner le bon kernel
1. Ouvrir ou créer un notebook
2. Cliquer sur le kernel en haut à droite  
3. Sélectionner **"Rakuten (Python 3.10.12)"**

### Vérifier l'environnement dans le notebook
```python
import sys
print(f"Python: {sys.executable}")
print(f"Dans virtualenv: {'venv' in sys.executable}")
```

## Compatibilité

- ✅ **CPU uniquement** : Fonctionne sur toutes les machines
- ✅ **NVIDIA GPU** : Accélération CUDA automatique si disponible
- ✅ **WSL2** : Support complet
- ✅ **Multi-plateforme** : Linux, Windows, macOS

## Structure des fichiers

```
projet/
├── requirements.txt      # Dépendances Python
├── install.sh           # Script d'installation Linux/macOS
├── install.bat          # Script d'installation Windows
└── README.md            # Ce fichier
```

## Dépannage

### Erreur "Python 3.10.x requis"
- Installer Python 3.10.12 depuis [python.org](https://python.org)
- Ou utiliser pyenv : `pyenv install 3.10.12`

### Erreur d'installation Pillow
```bash
# Ubuntu/Debian
sudo apt install python3-dev libjpeg-dev libpng-dev

# CentOS/RHEL
sudo yum install python3-devel libjpeg-devel libpng-devel
```

### Mode CPU forcé
Si CUDA n'est pas détecté mais que vous avez un GPU NVIDIA :
```bash
# Vérifier les drivers NVIDIA
nvidia-smi

# Réinstaller PyTorch avec CUDA
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2+cu117 --index-url https://download.pytorch.org/whl/cu117
```

### Kernel Jupyter non disponible
```bash
# Créer manuellement le kernel
pip install ipykernel
python -m ipykernel install --user --name=rakuten-3.10.12 --display-name="Rakuten (Python 3.10.12)"

# Lister les kernels disponibles
jupyter kernelspec list

# Supprimer un kernel si besoin
jupyter kernelspec remove rakuten-3.10.12
```

## Contact

Pour tout problème d'installation, vérifier :
1. Version Python : `python --version`
2. Pip à jour : `pip --version`
3. Espace disque suffisant : ~2GB requis