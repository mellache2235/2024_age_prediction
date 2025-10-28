# Environment Installation Guide for HPC

Complete instructions for setting up the conda environment on Stanford Menon Lab HPC at `/oak/stanford/groups/menon/software/python_envs/`

---

## 📋 Prerequisites

- Access to Stanford HPC (Sherlock/Oak)
- Conda/Miniconda installed
- Write access to `/oak/stanford/groups/menon/software/python_envs/`

---

## 🚀 Quick Install (Copy-Paste Ready)

```bash
# Navigate to shared environment directory
cd /oak/stanford/groups/menon/software/python_envs/

# Create new conda environment with Python 3.9
conda create -p /oak/stanford/groups/menon/software/python_envs/brain_age_2024 python=3.9 -y

# Activate environment
conda activate /oak/stanford/groups/menon/software/python_envs/brain_age_2024

# Install PyTorch FIRST (with CUDA 11.7 for GPU support)
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

# Install PyTorch Lightning and related packages
pip install pytorch-lightning==1.9.5

# Install core scientific packages
pip install numpy==1.23.5 pandas==1.5.3 scipy==1.10.1 scikit-learn==1.2.2

# Install visualization packages
pip install matplotlib==3.7.1 seaborn==0.12.2

# Install deep learning utilities
pip install tensorboard==2.13.0 einops==0.6.1 captum==0.6.0 timm==0.9.2

# Install data handling packages
pip install openpyxl==3.1.2 pyyaml==6.0 tabulate==0.9.0

# Install statistical packages
pip install statsmodels==0.14.0

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import numpy, pandas, scipy, sklearn, matplotlib, seaborn; print('All packages imported successfully!')"
```

---

## 📦 Complete Package List with Versions

### Core Scientific Computing
```
numpy==1.23.5
pandas==1.5.3
scipy==1.10.1
scikit-learn==1.2.2
statsmodels==0.14.0
```

### Deep Learning
```
torch==1.13.1+cu117
torchvision==0.14.1+cu117
torchaudio==0.13.1
pytorch-lightning==1.9.5
```

### Deep Learning Utilities
```
tensorboard==2.13.0
einops==0.6.1
captum==0.6.0
timm==0.9.2
```

### Visualization
```
matplotlib==3.7.1
seaborn==0.12.2
```

### Data Handling
```
openpyxl==3.1.2
pyyaml==6.0
tabulate==0.9.0
```

---

## 🔍 Import Verification Script

Save this as `verify_imports.py` and run to check all imports:

```python
#!/usr/bin/env python3
"""Verify all required packages are installed and importable."""

import sys

def check_imports():
    """Check all required imports."""
    packages = {
        # Core scientific
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'scipy': 'SciPy',
        'sklearn': 'Scikit-learn',
        'statsmodels': 'Statsmodels',
        
        # Deep learning
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'torchaudio': 'TorchAudio',
        'pytorch_lightning': 'PyTorch Lightning',
        
        # DL utilities
        'tensorboard': 'TensorBoard',
        'einops': 'Einops',
        'captum': 'Captum',
        'timm': 'TIMM',
        
        # Visualization
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        
        # Data handling
        'openpyxl': 'OpenPyXL',
        'yaml': 'PyYAML',
        'tabulate': 'Tabulate'
    }
    
    print("\n" + "="*60)
    print("CHECKING REQUIRED PACKAGES")
    print("="*60 + "\n")
    
    failed = []
    
    for module, name in packages.items():
        try:
            mod = __import__(module)
            version = getattr(mod, '__version__', 'unknown')
            print(f"✓ {name:20s} {version}")
        except ImportError as e:
            print(f"✗ {name:20s} FAILED: {e}")
            failed.append(name)
    
    # Special checks
    print("\n" + "="*60)
    print("SPECIAL CHECKS")
    print("="*60 + "\n")
    
    # Check CUDA
    try:
        import torch
        print(f"✓ PyTorch CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU count: {torch.cuda.device_count()}")
    except:
        print("✗ PyTorch CUDA check failed")
    
    # Check specific submodules
    try:
        from scipy.stats import spearmanr
        print("✓ scipy.stats.spearmanr")
    except:
        print("✗ scipy.stats.spearmanr FAILED")
        failed.append("scipy.stats")
    
    try:
        from sklearn.decomposition import PCA
        print("✓ sklearn.decomposition.PCA")
    except:
        print("✗ sklearn.decomposition.PCA FAILED")
        failed.append("sklearn.decomposition")
    
    try:
        from sklearn.linear_model import Ridge, Lasso, ElasticNet
        print("✓ sklearn.linear_model (Ridge, Lasso, ElasticNet)")
    except:
        print("✗ sklearn.linear_model FAILED")
        failed.append("sklearn.linear_model")
    
    try:
        from statsmodels.stats.multitest import multipletests
        print("✓ statsmodels.stats.multitest")
    except:
        print("✗ statsmodels.stats.multitest FAILED")
        failed.append("statsmodels.stats")
    
    try:
        import matplotlib.backends.backend_pdf
        print("✓ matplotlib.backends.backend_pdf")
    except:
        print("✗ matplotlib.backends.backend_pdf FAILED")
        failed.append("matplotlib.backends")
    
    # Summary
    print("\n" + "="*60)
    if failed:
        print(f"❌ FAILED: {len(failed)} package(s)")
        print(f"   {', '.join(failed)}")
        return False
    else:
        print("✅ SUCCESS: All packages installed correctly!")
        return True

if __name__ == "__main__":
    success = check_imports()
    sys.exit(0 if success else 1)
```

Run verification:
```bash
python verify_imports.py
```

---

## 🔧 Troubleshooting

### Issue 1: PyTorch CUDA not available

**Problem**: `torch.cuda.is_available()` returns `False`

**Solution**:
```bash
# Reinstall with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

### Issue 2: PyTorch Lightning compatibility

**Problem**: `ImportError: cannot import name 'LightningModule'`

**Solution**:
```bash
pip install pytorch-lightning==1.9.5 --force-reinstall
```

### Issue 3: NumPy version conflicts

**Problem**: `ImportError: numpy.core.multiarray failed to import`

**Solution**:
```bash
pip install numpy==1.23.5 --force-reinstall --no-cache-dir
```

### Issue 4: Pandas building from source

**Problem**: Pandas tries to build from source (very slow)

**Solution**:
```bash
pip install pandas==1.5.3 --only-binary :all:
```

### Issue 5: Matplotlib font issues

**Problem**: Arial font not found

**Solution**:
```bash
# Font is at HPC path - scripts will load it automatically
# Path: /oak/stanford/groups/menon/projects/mellache/2021_foundation_model/scripts/dnn/clustering_analysis/arial.ttf
# No action needed - scripts handle this
```

---

## 📝 Alternative: CPU-Only Installation

If you don't need GPU support:

```bash
# Create environment
conda create -p /oak/stanford/groups/menon/software/python_envs/brain_age_2024_cpu python=3.9 -y
conda activate /oak/stanford/groups/menon/software/python_envs/brain_age_2024_cpu

# Install PyTorch CPU version
pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cpu

# Install remaining packages (same as GPU version)
pip install pytorch-lightning==1.9.5
pip install numpy==1.23.5 pandas==1.5.3 scipy==1.10.1 scikit-learn==1.2.2
pip install matplotlib==3.7.1 seaborn==0.12.2
pip install tensorboard==2.13.0 einops==0.6.1 captum==0.6.0 timm==0.9.2
pip install openpyxl==3.1.2 pyyaml==6.0 tabulate==0.9.0 statsmodels==0.14.0
```

---

## 🎯 Using the Environment

### Activate
```bash
conda activate /oak/stanford/groups/menon/software/python_envs/brain_age_2024
```

### Run Scripts
```bash
cd /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/scripts
python run_network_analysis.py
python run_statistical_comparisons.py
python run_optimized_brain_behavior.py --all
```

### Deactivate
```bash
conda deactivate
```

---

## 📊 Environment Details

**Location**: `/oak/stanford/groups/menon/software/python_envs/brain_age_2024`  
**Python Version**: 3.9  
**Total Packages**: ~100 (including dependencies)  
**Disk Space**: ~3-4 GB  

---

## ✅ Compatibility Matrix

| Package | Version | Compatible With |
|---------|---------|----------------|
| Python | 3.9 | All packages |
| PyTorch | 1.13.1 | CUDA 11.7, Lightning 1.9.5 |
| NumPy | 1.23.5 | Pandas 1.5.3, SciPy 1.10.1 |
| Pandas | 1.5.3 | NumPy 1.23.5, Matplotlib 3.7.1 |
| Scikit-learn | 1.2.2 | NumPy 1.23.5, SciPy 1.10.1 |
| Matplotlib | 3.7.1 | NumPy 1.23.5, Seaborn 0.12.2 |
| PyTorch Lightning | 1.9.5 | PyTorch 1.13.1 |

**All versions tested for compatibility - no conflicts!**

---

## 🔗 SLURM Job Script Example

```bash
#!/bin/bash
#SBATCH --job-name=brain_age
#SBATCH --output=logs/brain_age_%j.out
#SBATCH --error=logs/brain_age_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=menon
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

# Load conda
source ~/.bashrc

# Activate environment
conda activate /oak/stanford/groups/menon/software/python_envs/brain_age_2024

# Run analysis
cd /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/scripts
python run_optimized_brain_behavior.py --all

# Deactivate
conda deactivate
```

---

## 📧 Support

If you encounter issues:
1. Check `verify_imports.py` output
2. Review error messages
3. Consult Troubleshooting section above
4. Contact Menon Lab HPC support

**Last Updated**: 2024  
**Tested On**: Stanford Sherlock/Oak HPC

