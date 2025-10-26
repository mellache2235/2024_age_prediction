#!/bin/bash
#
# Run brain-behavior analysis for ADHD200 TD cohort ONLY
#

echo "================================================================================================"
echo "🧠 BRAIN-BEHAVIOR ANALYSIS - ADHD200 TD"
echo "================================================================================================"
echo ""

# Full HPC paths
PKLZ_FILE="/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/data/imaging/for_dnn/adhd200_regression_age_TD_wIDs/fold_0.bin"
IG_CSV="/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/count_data/adhd200_td_count_data.csv"
OUTPUT_DIR="/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_behavior/adhd200_td"

echo "📂 Input files:"
echo "   PKLZ:   $PKLZ_FILE"
echo "   IG CSV: $IG_CSV"
echo "   Output: $OUTPUT_DIR"
echo ""
echo "ℹ️  Behavioral data (Hyper/Impulsive, Inattentive) is embedded in the .pklz file"
echo ""

# Run analysis
python brain_behavior_td_simple.py \
  --dataset adhd200_td \
  --pklz_file "$PKLZ_FILE" \
  --ig_csv "$IG_CSV" \
  --output_dir "$OUTPUT_DIR" \
  --n_components 10

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "================================================================================================"
    echo "✅ ADHD200 TD analysis complete!"
    echo "📊 Results saved to: $OUTPUT_DIR"
    echo "================================================================================================"
else
    echo "================================================================================================"
    echo "❌ ADHD200 TD analysis failed with exit code $EXIT_CODE"
    echo "================================================================================================"
fi

exit $EXIT_CODE

