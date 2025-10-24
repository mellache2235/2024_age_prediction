#!/usr/bin/env python3
"""
Automatic dependency installer for the age prediction analysis pipeline.

This script checks for required dependencies and installs missing packages.
"""

import subprocess
import sys
import importlib
from pathlib import Path
from typing import Optional


def check_package(package_name: str, import_name: Optional[str] = None) -> bool:
    """
    Check if a package is installed.
    
    Args:
        package_name (str): Package name for pip
        import_name (str, optional): Import name (if different from package name)
        
    Returns:
        bool: True if package is installed
    """
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False


def install_package(package_name: str) -> bool:
    """
    Install a package using pip.
    
    Args:
        package_name (str): Package name to install
        
    Returns:
        bool: True if installation successful
    """
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        return True
    except subprocess.CalledProcessError:
        return False


def main():
    """Main function to check and install dependencies."""
    print("Age Prediction Analysis Pipeline - Dependency Checker")
    print("=" * 60)
    
    # Required packages with their pip names and import names
    required_packages = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'scipy': 'scipy',
        'scikit-learn': 'sklearn',
        'torch': 'torch',
        'torchvision': 'torchvision',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'statsmodels': 'statsmodels',
        'nilearn': 'nilearn',
        'nibabel': 'nibabel',
        'captum': 'captum',
        'tqdm': 'tqdm',
        'pyyaml': 'yaml',
        'networkx': 'networkx',
        'joblib': 'joblib'
    }
    
    # Optional packages
    optional_packages = {
        'plotly': 'plotly',
        'loguru': 'loguru'
    }
    
    missing_required = []
    missing_optional = []
    
    print("Checking required packages...")
    for pip_name, import_name in required_packages.items():
        if check_package(pip_name, import_name):
            print(f"✓ {pip_name}")
        else:
            print(f"✗ {pip_name} (missing)")
            missing_required.append(pip_name)
    
    print("\nChecking optional packages...")
    for pip_name, import_name in optional_packages.items():
        if check_package(pip_name, import_name):
            print(f"✓ {pip_name}")
        else:
            print(f"✗ {pip_name} (missing)")
            missing_optional.append(pip_name)
    
    # Install missing required packages
    if missing_required:
        print(f"\nInstalling {len(missing_required)} missing required packages...")
        for package in missing_required:
            print(f"Installing {package}...")
            if install_package(package):
                print(f"✓ {package} installed successfully")
            else:
                print(f"✗ Failed to install {package}")
                print("Please install manually: pip install", package)
    
    # Install missing optional packages
    if missing_optional:
        print(f"\nInstalling {len(missing_optional)} missing optional packages...")
        for package in missing_optional:
            print(f"Installing {package}...")
            if install_package(package):
                print(f"✓ {package} installed successfully")
            else:
                print(f"✗ Failed to install {package} (optional)")
    
    # Final check
    print("\nFinal dependency check...")
    all_required_installed = True
    for pip_name, import_name in required_packages.items():
        if not check_package(pip_name, import_name):
            print(f"✗ {pip_name} still missing")
            all_required_installed = False
        else:
            print(f"✓ {pip_name}")
    
    if all_required_installed:
        print("\n🎉 All required dependencies are installed!")
        print("You can now run the age prediction analysis pipeline.")
    else:
        print("\n❌ Some required dependencies are still missing.")
        print("Please install them manually or check your Python environment.")
        sys.exit(1)


if __name__ == "__main__":
    main()
