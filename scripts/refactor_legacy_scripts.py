#!/usr/bin/env python3
"""
Script to help refactor legacy scripts to use the new modular structure.

This script provides utilities to identify and refactor existing scripts
to use the new utils modules and follow PEP8 standards.
"""

import os
import sys
import ast
import re
from pathlib import Path
from typing import List, Dict, Set, Tuple
import argparse


class ScriptRefactorer:
    """
    Class for refactoring legacy scripts to use the new modular structure.
    """
    
    def __init__(self, utils_path: str = "utils"):
        """
        Initialize ScriptRefactorer.
        
        Args:
            utils_path (str): Path to utils directory
        """
        self.utils_path = utils_path
        self.import_mappings = {
            'load_finetune_dataset': 'data_utils',
            'remove_nans': 'data_utils',
            'reshapeData': 'data_utils',
            'detect_roi_columns': 'data_utils',
            'ConvNet': 'model_utils',
            'RMSELoss': 'model_utils',
            'train_regressor_w_embedding': 'model_utils',
            'plot_age_prediction': 'plotting_utils',
            'plot_network_analysis': 'plotting_utils',
            'benjamini_hochberg_correction': 'statistical_utils',
            'multiple_correlation_analysis': 'statistical_utils',
            'compute_consensus_features_across_models': 'feature_utils'
        }
    
    def find_legacy_scripts(self, scripts_dir: str) -> List[str]:
        """
        Find legacy scripts that need refactoring.
        
        Args:
            scripts_dir (str): Directory containing scripts
            
        Returns:
            List[str]: List of script paths
        """
        legacy_scripts = []
        
        for root, dirs, files in os.walk(scripts_dir):
            # Skip certain directories
            if any(skip_dir in root for skip_dir in ['__pycache__', '.git', 'results']):
                continue
            
            for file in files:
                if file.endswith('.py') and not file.startswith('__'):
                    script_path = os.path.join(root, file)
                    
                    # Check if script uses legacy patterns
                    if self._is_legacy_script(script_path):
                        legacy_scripts.append(script_path)
        
        return legacy_scripts
    
    def _is_legacy_script(self, script_path: str) -> bool:
        """
        Check if a script is a legacy script that needs refactoring.
        
        Args:
            script_path (str): Path to script
            
        Returns:
            bool: True if script is legacy
        """
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for legacy patterns
            legacy_patterns = [
                r'from utility_functions import',
                r'import utility_functions',
                r'def load_finetune_dataset',
                r'def remove_nans',
                r'class ConvNet',
                r'def plot_ages',
                r'def train_Regressor_wEmbedding'
            ]
            
            for pattern in legacy_patterns:
                if re.search(pattern, content):
                    return True
            
            return False
            
        except Exception as e:
            print(f"Error reading {script_path}: {e}")
            return False
    
    def analyze_script_imports(self, script_path: str) -> Dict[str, List[str]]:
        """
        Analyze imports in a script.
        
        Args:
            script_path (str): Path to script
            
        Returns:
            Dict[str, List[str]]: Dictionary of import analysis
        """
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            imports = {
                'standard_imports': [],
                'third_party_imports': [],
                'local_imports': [],
                'legacy_imports': []
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.startswith('utility_functions'):
                            imports['legacy_imports'].append(alias.name)
                        elif alias.name in ['numpy', 'pandas', 'matplotlib', 'seaborn']:
                            imports['third_party_imports'].append(alias.name)
                        else:
                            imports['standard_imports'].append(alias.name)
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.module.startswith('utility_functions'):
                        imports['legacy_imports'].append(node.module)
                    elif node.module and node.module in ['numpy', 'pandas', 'matplotlib', 'seaborn']:
                        imports['third_party_imports'].append(node.module)
                    elif node.module and not node.module.startswith('.'):
                        imports['local_imports'].append(node.module)
            
            return imports
            
        except Exception as e:
            print(f"Error analyzing imports in {script_path}: {e}")
            return {}
    
    def generate_refactoring_suggestions(self, script_path: str) -> List[str]:
        """
        Generate refactoring suggestions for a script.
        
        Args:
            script_path (str): Path to script
            
        Returns:
            List[str]: List of refactoring suggestions
        """
        suggestions = []
        
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for legacy imports
            if 'utility_functions' in content:
                suggestions.append("Replace 'utility_functions' imports with specific utils modules")
            
            # Check for legacy function definitions
            legacy_functions = [
                'load_finetune_dataset', 'remove_nans', 'reshapeData',
                'ConvNet', 'RMSELoss', 'plot_ages', 'train_Regressor_wEmbedding'
            ]
            
            for func in legacy_functions:
                if f'def {func}' in content:
                    suggestions.append(f"Remove local definition of '{func}' and import from utils")
            
            # Check for PEP8 violations
            if re.search(r'[a-z][A-Z]', content):
                suggestions.append("Fix camelCase variable names to snake_case")
            
            if re.search(r'^[^#]*\t', content, re.MULTILINE):
                suggestions.append("Replace tabs with spaces for indentation")
            
            # Check for hardcoded paths
            if re.search(r'["\'][/\\\\][^"\']*["\']', content):
                suggestions.append("Replace hardcoded paths with configurable parameters")
            
            # Check for missing docstrings
            if 'def ' in content and '"""' not in content:
                suggestions.append("Add docstrings to functions")
            
            return suggestions
            
        except Exception as e:
            print(f"Error generating suggestions for {script_path}: {e}")
            return []
    
    def create_refactoring_report(self, scripts_dir: str, output_path: str = "refactoring_report.txt") -> None:
        """
        Create a comprehensive refactoring report.
        
        Args:
            scripts_dir (str): Directory containing scripts
            output_path (str): Output path for report
        """
        legacy_scripts = self.find_legacy_scripts(scripts_dir)
        
        with open(output_path, 'w') as f:
            f.write("Age Prediction Analysis - Refactoring Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Found {len(legacy_scripts)} legacy scripts that need refactoring:\n\n")
            
            for script_path in legacy_scripts:
                f.write(f"Script: {script_path}\n")
                f.write("-" * len(script_path) + "\n")
                
                # Analyze imports
                imports = self.analyze_script_imports(script_path)
                if imports:
                    f.write("Current imports:\n")
                    for import_type, import_list in imports.items():
                        if import_list:
                            f.write(f"  {import_type}: {', '.join(import_list)}\n")
                    f.write("\n")
                
                # Generate suggestions
                suggestions = self.generate_refactoring_suggestions(script_path)
                if suggestions:
                    f.write("Refactoring suggestions:\n")
                    for i, suggestion in enumerate(suggestions, 1):
                        f.write(f"  {i}. {suggestion}\n")
                    f.write("\n")
                
                f.write("\n")
            
            f.write("Refactoring Guidelines:\n")
            f.write("-" * 25 + "\n")
            f.write("1. Replace 'utility_functions' imports with specific utils modules\n")
            f.write("2. Remove duplicate function definitions and import from utils\n")
            f.write("3. Follow PEP8 naming conventions (snake_case for variables/functions)\n")
            f.write("4. Add comprehensive docstrings to all functions\n")
            f.write("5. Replace hardcoded paths with configurable parameters\n")
            f.write("6. Use type hints for function parameters and returns\n")
            f.write("7. Organize imports: standard library, third-party, local\n")
            f.write("8. Add error handling and logging where appropriate\n")
        
        print(f"Refactoring report saved to: {output_path}")


def main():
    """Main function for script refactoring."""
    parser = argparse.ArgumentParser(
        description="Analyze and refactor legacy scripts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all scripts and create refactoring report
  python refactor_legacy_scripts.py --scripts_dir scripts --create_report

  # Analyze specific script
  python refactor_legacy_scripts.py --scripts_dir scripts --script_path scripts/legacy_script.py
        """
    )
    
    parser.add_argument("--scripts_dir", type=str, default="scripts",
                       help="Directory containing scripts to analyze")
    parser.add_argument("--script_path", type=str,
                       help="Specific script to analyze")
    parser.add_argument("--create_report", action='store_true',
                       help="Create comprehensive refactoring report")
    parser.add_argument("--output_report", type=str, default="refactoring_report.txt",
                       help="Output path for refactoring report")
    
    args = parser.parse_args()
    
    refactorer = ScriptRefactorer()
    
    if args.script_path:
        # Analyze specific script
        if not os.path.exists(args.script_path):
            print(f"Error: Script not found: {args.script_path}")
            sys.exit(1)
        
        print(f"Analyzing script: {args.script_path}")
        
        # Analyze imports
        imports = refactorer.analyze_script_imports(args.script_path)
        print("Import analysis:")
        for import_type, import_list in imports.items():
            if import_list:
                print(f"  {import_type}: {', '.join(import_list)}")
        
        # Generate suggestions
        suggestions = refactorer.generate_refactoring_suggestions(args.script_path)
        print("\nRefactoring suggestions:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"  {i}. {suggestion}")
    
    else:
        # Analyze all scripts
        if not os.path.exists(args.scripts_dir):
            print(f"Error: Scripts directory not found: {args.scripts_dir}")
            sys.exit(1)
        
        legacy_scripts = refactorer.find_legacy_scripts(args.scripts_dir)
        print(f"Found {len(legacy_scripts)} legacy scripts:")
        for script in legacy_scripts:
            print(f"  - {script}")
        
        if args.create_report:
            refactorer.create_refactoring_report(args.scripts_dir, args.output_report)


if __name__ == "__main__":
    main()
