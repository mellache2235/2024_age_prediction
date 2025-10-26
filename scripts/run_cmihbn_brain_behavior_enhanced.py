#!/usr/bin/env python3
"""
Run enhanced brain-behavior analysis for CMI-HBN TD with all paths pre-configured.

Just run: python run_cmihbn_brain_behavior_enhanced.py
"""

import subprocess
import sys

# All paths pre-configured (no arguments needed)
DATASET = "cmihbn_td"
PKLZ_DIR = "/oak/stanford/groups/menon/deriveddata/public/cmihbn/restfmri/timeseries/group_level/brainnetome/normz"
IG_CSV = "/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/integrated_gradients/cmihbn_td_features_all_sites_IG_convnet_regressor_trained_on_hcp_dev_top_regions_wIDS_single_model_pred.csv"
C3SR_DIR = "/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/prepare_data/cmihbn/behavior"
OUTPUT_DIR = "/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_behavior/cmihbn_td"

print("="*100)
print("🧠 ENHANCED BRAIN-BEHAVIOR ANALYSIS - CMI-HBN TD")
print("="*100)
print()
print("📂 Configuration:")
print(f"   PKLZ DIR:   {PKLZ_DIR}")
print(f"   IG CSV:     {IG_CSV}")
print(f"   C3SR DIR:   {C3SR_DIR}")
print(f"   Output:     {OUTPUT_DIR}")
print()

# Run the enhanced script
cmd = [
    sys.executable,
    "brain_behavior_enhanced.py",
    "--pklz_file", PKLZ_DIR,
    "--ig_csv", IG_CSV,
    "--c3sr_dir", C3SR_DIR,
    "--output_dir", OUTPUT_DIR,
    "--dataset", DATASET
]

try:
    result = subprocess.run(cmd, check=True)
    print()
    print("="*100)
    print("✅ CMI-HBN TD Enhanced Analysis Complete!")
    print(f"📊 Results saved to: {OUTPUT_DIR}")
    print("="*100)
    sys.exit(0)
except subprocess.CalledProcessError as e:
    print()
    print("="*100)
    print(f"❌ Analysis failed with exit code {e.returncode}")
    print("="*100)
    sys.exit(1)

