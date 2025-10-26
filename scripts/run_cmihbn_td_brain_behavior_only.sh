#!/bin/bash
#
# Run brain-behavior analysis for CMI-HBN TD cohort ONLY
#

echo "================================================================================================"
echo "🧠 BRAIN-BEHAVIOR ANALYSIS - CMI-HBN TD"
echo "================================================================================================"
echo ""

# Full HPC paths
PKLZ_FILE="/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/data/imaging/for_dnn/cmihbn_age_TD/fold_0.bin"
IG_CSV="/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/count_data/cmihbn_td_count_data.csv"
C3SR_CSV="/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/prepare_data/cmihbn/behavior"
OUTPUT_DIR="/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_behavior/cmihbn_td"

echo "📂 Input files:"
echo "   PKLZ:   $PKLZ_FILE"
echo "   IG CSV: $IG_CSV"
echo "   C3SR:   $C3SR_CSV (directory - will auto-detect Conners CSV with C3SR_HY_T, C3SR_IN_T)"
echo "   Output: $OUTPUT_DIR"
echo ""
echo "ℹ️  Script will look for Conners CSV files with C3SR_HY_T and C3SR_IN_T columns"
echo ""

# Run analysis
python brain_behavior_td_simple.py \
  --dataset cmihbn_td \
  --pklz_file "$PKLZ_FILE" \
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

