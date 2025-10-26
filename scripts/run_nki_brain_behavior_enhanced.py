#!/usr/bin/env python3
"""
Run enhanced brain-behavior analysis for NKI-RS TD with all paths pre-configured.

Just run: python run_nki_brain_behavior_enhanced.py
"""

import subprocess
import sys

# All paths pre-configured (no arguments needed)
DATASET = "nki_rs_td"
IG_CSV = "/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/integrated_gradients/nki_cog_dev_wIDS_features_IG_convnet_regressor_single_model_fold_0.csv"
BEHAVIORAL_FILE = "/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/scripts/FLUX/assessment_data/8100_CAARS-S-S_20191009.csv"
OUTPUT_DIR = "/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_behavior/nki_rs_td"

print("="*100)
print("🧠 ENHANCED BRAIN-BEHAVIOR ANALYSIS - NKI-RS TD")
print("="*100)
print()
print("📂 Configuration:")
print(f"   IG CSV:     {IG_CSV}")
print(f"   CAARS CSV:  {BEHAVIORAL_FILE}")
print(f"   Output:     {OUTPUT_DIR}")
print()

# Run the enhanced script
cmd = [
    sys.executable,
    "brain_behavior_enhanced.py",
    "--ig_csv", IG_CSV,
    "--behavioral_file", BEHAVIORAL_FILE,
    "--output_dir", OUTPUT_DIR,
    "--dataset", DATASET
]

try:
    result = subprocess.run(cmd, check=True)
    print()
    print("="*100)
    print("✅ NKI-RS TD Enhanced Analysis Complete!")
    print(f"📊 Results saved to: {OUTPUT_DIR}")
    print("="*100)
    sys.exit(0)
except subprocess.CalledProcessError as e:
    print()
    print("="*100)
    print(f"❌ Analysis failed with exit code {e.returncode}")
    print("="*100)
    sys.exit(1)

