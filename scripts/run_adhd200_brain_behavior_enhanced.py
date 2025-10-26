#!/usr/bin/env python3
"""
Run enhanced brain-behavior analysis for ADHD200 TD with all paths pre-configured.

Just run: python run_adhd200_brain_behavior_enhanced.py
"""

import subprocess
import sys

# All paths pre-configured (no arguments needed)
DATASET = "adhd200_td"
PKLZ_FILE = "/oak/stanford/groups/menon/deriveddata/public/adhd200/restfmri/timeseries/group_level/brainnetome/normz/adhd200_run-rest_brainnetome_mean_regMov-6param_wmcsf_dt1_bpf008-09_normz_246ROIs.pklz"
IG_CSV = "/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/integrated_gradients/adhd200_td_features_all_sites_IG_convnet_regressor_trained_on_hcp_dev_top_regions_wIDS_single_model_pred.csv"
OUTPUT_DIR = "/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_behavior/adhd200_td"

print("="*100)
print("🧠 ENHANCED BRAIN-BEHAVIOR ANALYSIS - ADHD200 TD")
print("="*100)
print()
print("📂 Configuration:")
print(f"   PKLZ File:  {PKLZ_FILE}")
print(f"   IG CSV:     {IG_CSV}")
print(f"   Output:     {OUTPUT_DIR}")
print()

# Run the enhanced script
cmd = [
    sys.executable,
    "brain_behavior_enhanced.py",
    "--pklz_file", PKLZ_FILE,
    "--ig_csv", IG_CSV,
    "--output_dir", OUTPUT_DIR,
    "--dataset", DATASET
]

try:
    result = subprocess.run(cmd, check=True)
    print()
    print("="*100)
    print("✅ ADHD200 TD Enhanced Analysis Complete!")
    print(f"📊 Results saved to: {OUTPUT_DIR}")
    print("="*100)
    sys.exit(0)
except subprocess.CalledProcessError as e:
    print()
    print("="*100)
    print(f"❌ Analysis failed with exit code {e.returncode}")
    print("="*100)
    sys.exit(1)

