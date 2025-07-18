@echo off
REM Installation automatique du projet Rakuten
REM Compatible Python 3.10.12

echo 🚀 Installation du projet Rakuten...

REM Vérifier la version Python
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set python_version=%%i
echo Version Python détectée: %python_version%

REM Vérifier si c'est Python 3.10.x
echo %python_version% | findstr /C:"3.10." >nul
if errorlevel 1 (
    echo ❌ Erreur: Python 3.10.x requis, version détectée: %python_version%
    echo 💡 Installer Python 3.10.12 depuis python.org ou avec pyenv
    pause
    exit /b 1
)

echo ✅ Python %python_version% détecté

REM Mettre à jour pip
echo 📦 Mise à jour de pip...
pip install --upgrade pip setuptools wheel
if errorlevel 1 (
    echo ❌ Erreur lors de la mise à jour de pip
    pause
    exit /b 1
)

REM Installer PyTorch avec CUDA si possible
echo 🔥 Installation de PyTorch avec CUDA...
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2+cu117 --index-url https://download.pytorch.org/whl/cu117 >nul 2>&1

REM Si l'installation CUDA échoue, installer la version CPU
if errorlevel 1 (
    echo ⚠️  Installation CUDA échouée, installation version CPU...
    pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
    if errorlevel 1 (
        echo ❌ Erreur lors de l'installation de PyTorch
        pause
        exit /b 1
    )
)

REM Installer le reste des dépendances
echo 📚 Installation des autres dépendances...
pip install -r requirements.txt
if errorlevel 1 (
    echo ❌ Erreur lors de l'installation des dépendances
    pause
    exit /b 1
)

REM Installer et configurer le kernel Jupyter
echo 🔧 Configuration du kernel Jupyter...
pip install ipykernel
if errorlevel 1 (
    echo ❌ Erreur lors de l'installation d'ipykernel
    pause
    exit /b 1
)

python -m ipykernel install --user --name=rakuten-3.10.12 --display-name="Rakuten (Python 3.10.12)"
if errorlevel 1 (
    echo ❌ Erreur lors de la création du kernel
    pause
    exit /b 1
)

REM Test de l'installation
echo 🧪 Test de l'installation...
python -c "import torch; print(f'✅ PyTorch {torch.__version__} installé'); print(f'🎮 CUDA disponible: {torch.cuda.is_available()}'); print(f'🔧 GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else '💻 Mode CPU activé')"

if errorlevel 1 (
    echo ❌ Erreur lors du test de l'installation
    pause
    exit /b 1
)

echo.
echo 📓 Kernel Jupyter 'Rakuten (Python 3.10.12)' créé
echo 🎉 Installation terminée ! Le projet est prêt à être utilisé.
echo.
echo 🚀 Pour démarrer Jupyter Lab:
echo    jupyter lab
echo.
pause