#!/bin/bash
# Installation automatique du projet Rakuten
# Compatible Python 3.10.12

echo "🚀 Installation du projet Rakuten..."

# Vérifier la version Python
python_version=$(python --version 2>&1 | grep -o '[0-9]\+\.[0-9]\+\.[0-9]\+')
if [[ ! "$python_version" =~ ^3\.10\. ]]; then
    echo "❌ Erreur: Python 3.10.x requis, version détectée: $python_version"
    echo "💡 Installer Python 3.10.12 avec pyenv ou votre gestionnaire de versions"
    exit 1
fi

echo "✅ Python $python_version détecté"

# Mettre à jour pip
echo "📦 Mise à jour de pip..."
pip install --upgrade pip setuptools wheel

# Installer PyTorch avec CUDA si possible, sinon CPU
echo "🔥 Installation de PyTorch..."
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2+cu117 --index-url https://download.pytorch.org/whl/cu117 2>/dev/null

# Si l'installation CUDA échoue, installer la version CPU
if [ $? -ne 0 ]; then
    echo "⚠️  Installation CUDA échouée, installation version CPU..."
    pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
fi

# Installer le reste des dépendances
echo "📚 Installation des autres dépendances..."
pip install -r requirements.txt

# Installer et configurer le kernel Jupyter
echo "🔧 Configuration du kernel Jupyter..."
pip install ipykernel
python -m ipykernel install --user --name=rakuten-3.10.12 --display-name="Rakuten (Python 3.10.12)"

# Test de l'installation
echo "🧪 Test de l'installation..."
python -c "
import torch
print(f'✅ PyTorch {torch.__version__} installé')
print(f'🎮 CUDA disponible: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'🔧 GPU: {torch.cuda.get_device_name(0)}')
else:
    print('💻 Mode CPU activé')
"

echo "📓 Kernel Jupyter 'Rakuten (Python 3.10.12)' créé"
echo "🎉 Installation terminée ! Le projet est prêt à être utilisé."
echo ""
echo "🚀 Pour démarrer Jupyter Lab:"
echo "   jupyter lab"