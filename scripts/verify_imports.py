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

