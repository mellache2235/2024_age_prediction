#!/bin/bash
#
# Run brain-behavior analysis for CMI-HBN TD cohort ONLY
#

echo "================================================================================================"
echo "🧠 BRAIN-BEHAVIOR ANALYSIS - CMI-HBN TD"
echo "================================================================================================"
echo ""

# Full HPC paths
PKLZ_DIR="/oak/stanford/groups/menon/deriveddata/public/cmihbn/restfmri/timeseries/group_level/brainnetome/normz"
IG_CSV="/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/count_data/cmihbn_td_count_data.csv"
C3SR_CSV="/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/prepare_data/cmihbn/behavior"
OUTPUT_DIR="/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_behavior/cmihbn_td"

echo "📂 Input files:"
echo "   PKLZ DIR: $PKLZ_DIR (will load all run1 files)"
echo "   IG CSV:   $IG_CSV"
echo "   C3SR:     $C3SR_CSV (directory - will auto-detect Conners CSV with C3SR_HY_T, C3SR_IN_T)"
echo "   Output:   $OUTPUT_DIR"
echo ""
echo "ℹ️  Script will:"
echo "   • Load all run1 .pklz files from directory"
echo "   • Filter for TD subjects (label != 99, label == 0)"
echo "   • Filter for quality (mean_fd < 0.5)"
echo "   • Merge with C3SR behavioral data"
echo ""

# Run analysis
python brain_behavior_td_simple.py \
  --dataset cmihbn_td \
  --pklz_file "$PKLZ_DIR" \
  --ig_csv "$IG_CSV" \
  --output_dir "$OUTPUT_DIR" \
  --n_components 10

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "================================================================================================"
    echo "✅ CMI-HBN TD analysis complete!"
    echo "📊 Results saved to: $OUTPUT_DIR"
    echo "================================================================================================"
else
    echo "================================================================================================"
    echo "❌ CMI-HBN TD analysis failed with exit code $EXIT_CODE"
    echo "================================================================================================"
fi

exit $EXIT_CODE

