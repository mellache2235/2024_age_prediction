#!/usr/bin/env python3
"""
Run brain-behavior analysis for both CMI-HBN TD and ADHD200 TD cohorts.

This script runs the brain-behavior PCA analysis for both TD cohorts sequentially.
"""

import subprocess
import sys
from pathlib import Path

# Add utils to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
utils_path = str(project_root / 'utils')
if utils_path not in sys.path:
    sys.path.append(utils_path)

from logging_utils import print_section_header, print_step, print_success, print_error, print_completion


def run_analysis(dataset_name, pklz_file, ig_csv, output_dir, n_components=10):
    """Run brain-behavior analysis for a single dataset."""
    
    print_step("", f"RUNNING {dataset_name.upper()}", f"PCA with {n_components} components")
    
    cmd = [
        sys.executable,
        str(script_dir / "brain_behavior_td_simple.py"),
        "--dataset", dataset_name,
        "--pklz_file", pklz_file,
        "--ig_csv", ig_csv,
        "--output_dir", output_dir,
        "--n_components", str(n_components)
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print_success(f"{dataset_name.upper()} analysis completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"{dataset_name.upper()} analysis failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print_error(f"Error running {dataset_name.upper()} analysis: {e}")
        return False


def main():
    print_section_header("BRAIN-BEHAVIOR ANALYSIS - BOTH TD COHORTS")
    
    # Configuration
    base_data_dir = "/oak/stanford/groups/menon/projects/wdcai/2019_ADHD_NN/data/imaging/timeseries"
    base_results_dir = "/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results"
    
    datasets = {
        "cmihbn_td": {
            "pklz": f"{base_data_dir}/CMIHBN/restfmri/timeseries/group_level/brainnetome/normz/cmihbn_run-rest_brainnetome_mean_regMov-6param_wmcsf_dt1_bpf008-09_normz_246ROIs_nn.pklz",
            "ig_csv": f"{base_results_dir}/integrated_gradients/cmihbn_td_ig_scores.csv",
            "output": f"{base_results_dir}/brain_behavior/cmihbn_td"
        },
        "adhd200_td": {
            "pklz": f"{base_data_dir}/ADHD200/restfmri/timeseries/group_level/brainnetome/normz/adhd200_run-rest_brainnetome_mean_regMov-6param_wmcsf_dt1_bpf008-09_normz_246ROIs_nn.pklz",
            "ig_csv": f"{base_results_dir}/integrated_gradients/adhd200_td_ig_scores.csv",
            "output": f"{base_results_dir}/brain_behavior/adhd200_td"
        }
    }
    
    n_components = 10
    
    # Run analyses
    results = {}
    all_output_files = []
    
    for dataset_name, config in datasets.items():
        print()
        success = run_analysis(
            dataset_name=dataset_name,
            pklz_file=config["pklz"],
            ig_csv=config["ig_csv"],
            output_dir=config["output"],
            n_components=n_components
        )
        results[dataset_name] = success
        
        if success:
            # Add expected output files
            output_dir = Path(config["output"])
            all_output_files.extend([
                str(output_dir / f"{dataset_name}_HY_pca_correlations.csv"),
                str(output_dir / f"{dataset_name}_IN_pca_correlations.csv")
            ])
        print()
    
    # Summary
    print_section_header("ANALYSIS SUMMARY")
    
    for dataset_name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {dataset_name.upper()}: {status}")
    
    print()
    
    if all(results.values()):
        print_completion("Both TD Cohorts Brain-Behavior Analysis", all_output_files)
        return 0
    else:
        print_error("Some analyses failed. Check output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

