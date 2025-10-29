#!/usr/bin/env python3
"""
Quick test script to verify the optimized brain-behavior analysis works correctly.
Tests imports, data loading, and basic functionality.
"""

import sys
from pathlib import Path

# Add to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'utils'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

def test_imports():
    """Test that all required imports work."""
    print("Testing imports...")
    try:
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from scipy.stats import spearmanr
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.model_selection import cross_val_score, KFold
        from sklearn.metrics import mean_absolute_error, r2_score, make_scorer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
        from joblib import Parallel, delayed
        
        from logging_utils import (print_section_header, print_step, print_success, 
                                   print_warning, print_error, print_info, print_completion)
        from plot_styles import create_standardized_scatter, get_dataset_title, setup_arial_font, DPI
        
        print("  ✓ All imports successful")
        return True
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False


def test_helper_functions():
    """Test helper functions from the optimized script."""
    print("\nTesting helper functions...")
    try:
        import numpy as np
        import pandas as pd
        
        # Import functions from the optimized script
        import run_stanford_asd_brain_behavior_optimized as opt_script
        
        # Test check_data_integrity
        test_df = pd.DataFrame({
            'subject_id': ['1', '2', '3'],
            'feature1': [1.0, 2.0, 3.0],
            'feature2': [4.0, 5.0, 6.0]
        })
        opt_script.check_data_integrity(test_df, "Test DataFrame")
        
        # Test determine_pc_range
        pc_range = opt_script.determine_pc_range(100, 200)
        assert len(pc_range) > 0, "PC range should not be empty"
        
        # Test spearman_scorer
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.2, 2.9, 4.1, 5.0])
        score = opt_script.spearman_scorer(y_true, y_pred)
        assert -1 <= score <= 1, "Spearman score should be between -1 and 1"
        
        print("  ✓ All helper functions working correctly")
        return True
    except Exception as e:
        print(f"  ✗ Helper function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_optimization_logic():
    """Test optimization logic with synthetic data."""
    print("\nTesting optimization logic with synthetic data...")
    try:
        import numpy as np
        from sklearn.model_selection import KFold
        from sklearn.metrics import make_scorer
        from scipy.stats import spearmanr
        
        import run_stanford_asd_brain_behavior_optimized as opt_script
        
        # Create synthetic data
        np.random.seed(42)
        n_samples = 50
        n_features = 30
        
        X = np.random.randn(n_samples, n_features)
        y = X[:, 0] + X[:, 1] * 0.5 + np.random.randn(n_samples) * 0.5  # Linear relationship
        
        # Modify the script's settings for faster testing
        opt_script.MAX_N_PCS = 10
        opt_script.PC_STEP = 5
        opt_script.MAX_PLS_COMPONENTS = 10
        opt_script.PLS_STEP = 5
        opt_script.ALPHA_RANGE = [0.1, 1.0]
        opt_script.TOP_K_FEATURES = [10, 20]
        opt_script.OUTER_CV_FOLDS = 3  # Reduce CV folds for speed
        
        # Test comprehensive optimization
        best_model, best_params, cv_score, results_df = opt_script.optimize_comprehensive(
            X, y, "test_measure", n_jobs=1
        )
        
        # Verify results
        assert best_model is not None, "Best model should not be None"
        assert best_params is not None, "Best params should not be None"
        assert cv_score is not None, "CV score should not be None"
        assert results_df is not None, "Results dataframe should not be None"
        assert len(results_df) > 0, "Results dataframe should not be empty"
        assert 'mean_cv_spearman' in results_df.columns, "Results should contain mean_cv_spearman"
        
        print(f"  ✓ Optimization completed successfully")
        print(f"    Best strategy: {best_params.get('strategy', 'N/A')}")
        print(f"    Best model: {best_params.get('model', 'N/A')}")
        print(f"    CV Spearman: {cv_score:.4f}")
        print(f"    Tested {len(results_df)} configurations")
        
        return True
    except Exception as e:
        print(f"  ✗ Optimization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*80)
    print("TESTING OPTIMIZED BRAIN-BEHAVIOR ANALYSIS SCRIPT")
    print("="*80)
    
    results = {
        'Imports': test_imports(),
        'Helper Functions': test_helper_functions(),
        'Optimization Logic': test_optimization_logic()
    }
    
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:.<50} {status}")
    
    all_passed = all(results.values())
    
    print("="*80)
    if all_passed:
        print("✓ ALL TESTS PASSED - Script is ready to use!")
    else:
        print("✗ SOME TESTS FAILED - Please review errors above")
    print("="*80)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

