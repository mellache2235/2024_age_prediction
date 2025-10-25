#!/usr/bin/env python3
"""
Run the complete analysis pipeline with beautiful, clear output.

This script provides a user-friendly interface for running all analysis steps
with clear progress indicators and formatted output.
"""

import os
import sys
import subprocess
from pathlib import Path
import time

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent / 'utils'))

from logging_utils import (print_section_header, print_step, print_success, 
                           print_warning, print_error, print_info, 
                           print_completion)


def run_command(command: str, description: str, show_output: bool = True):
    """
    Run a command with nice formatting.
    
    Args:
        command (str): Command to run
        description (str): Description of what the command does
        show_output (bool): Whether to show command output
    
    Returns:
        bool: True if successful, False otherwise
    """
    print_info(f"Running: {description}")
    print_info(f"Command: {command}", indent=2)
    print()
    
    start_time = time.time()
    
    try:
        if show_output:
            result = subprocess.run(command, shell=True, check=True)
        else:
            result = subprocess.run(command, shell=True, check=True, 
                                   capture_output=True, text=True)
        
        elapsed = time.time() - start_time
        print_success(f"Completed in {elapsed:.1f}s")
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print_error(f"Failed after {elapsed:.1f}s")
        if not show_output and e.stderr:
            print_error(f"Error: {e.stderr}")
        return False


def main():
    """Run the complete pipeline."""
    
    print_section_header("AGE PREDICTION ANALYSIS PIPELINE")
    
    # Define base paths
    base_dir = "/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test"
    scripts_dir = f"{base_dir}/scripts"
    results_dir = f"{base_dir}/results"
    
    print_info(f"Base Directory: {base_dir}")
    print_info(f"Results Directory: {results_dir}")
    print()
    
    # Change to scripts directory
    os.chdir(scripts_dir)
    
    # Step 1: Convert count data
    print_step(1, "CONVERT COUNT DATA", "Converting Excel files to CSV format")
    success = run_command(
        "python convert_count_data.py",
        "Converting Excel count data to CSV"
    )
    if not success:
        print_error("Step 1 failed. Aborting pipeline.")
        return
    print()
    
    # Step 2: Individual network analysis
    print_step(2, "INDIVIDUAL NETWORK ANALYSIS", "Creating radar plots for each dataset")
    success = run_command(
        "python network_analysis_yeo.py --process_all",
        "Processing all individual datasets"
    )
    if not success:
        print_warning("Step 2 had issues. Continuing...")
    print()
    
    # Step 3: Shared network analysis
    print_step(3, "SHARED NETWORK ANALYSIS", "Creating shared cohort radar plots (TD, ADHD, ASD)")
    success = run_command(
        "python network_analysis_yeo.py --process_shared",
        "Processing shared network analysis"
    )
    if not success:
        print_warning("Step 3 had issues. Continuing...")
    print()
    
    # Step 4: Create region tables
    print_step(4, "CREATE REGION TABLES", "Generating CSV tables with brain regions")
    success = run_command(
        f"python create_region_tables.py --config {base_dir}/config.yaml --output_dir {results_dir}/region_tables",
        "Creating region tables"
    )
    if not success:
        print_warning("Step 4 had issues. Continuing...")
    print()
    
    # Step 5: Brain age plots
    print_step(5, "BRAIN AGE PREDICTION PLOTS", "Generating scatter plots for all cohorts")
    
    print_info("5a. TD Cohorts (2x2 layout)")
    success = run_command(
        f"python plot_brain_age_td_cohorts.py --output_dir {results_dir}/brain_age_plots",
        "TD cohorts plot"
    )
    if not success:
        print_warning("TD cohorts plot had issues.")
    print()
    
    print_info("5b. ADHD Cohorts (1x2 layout)")
    success = run_command(
        f"python plot_brain_age_adhd_cohorts.py --output_dir {results_dir}/brain_age_plots",
        "ADHD cohorts plot"
    )
    if not success:
        print_warning("ADHD cohorts plot had issues.")
    print()
    
    print_info("5c. ASD Cohorts (1x2 layout)")
    success = run_command(
        f"python plot_brain_age_asd_cohorts.py --output_dir {results_dir}/brain_age_plots",
        "ASD cohorts plot"
    )
    if not success:
        print_warning("ASD cohorts plot had issues.")
    print()
    
    # Summary of output files
    output_files = [
        f"{results_dir}/count_data/*.csv",
        f"{results_dir}/network_analysis_yeo/*/",
        f"{results_dir}/region_tables/*.csv",
        f"{results_dir}/brain_age_plots/*.png",
    ]
    
    print_completion("Age Prediction Analysis Pipeline", output_files)
    
    print_info("To download PNG files from HPC:")
    print_info(f"  scp -r username@login.sherlock.stanford.edu:{results_dir}/*.png ./local_folder/", indent=2)
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_error("\n\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"\n\nPipeline failed with error: {e}")
        sys.exit(1)

