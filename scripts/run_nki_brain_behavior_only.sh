#!/bin/bash
#
# Run brain-behavior analysis for NKI-RS TD cohort ONLY
#

echo "================================================================================================"
echo "🧠 BRAIN-BEHAVIOR ANALYSIS - NKI-RS TD"
echo "================================================================================================"
echo ""

# Full HPC paths
IG_CSV="/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/integrated_gradients/nki_cog_dev_wIDS_features_IG_convnet_regressor_single_model_fold_0.csv"
CAARS_FILE="/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/scripts/FLUX/assessment_data/8100_CAARS-S-S_20191009.csv"
OUTPUT_DIR="/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_behavior/nki_rs_td"

echo "📂 Input files:"
echo "   IG CSV: $IG_CSV"
echo "   CAARS:  $CAARS_FILE"
echo "   Output: $OUTPUT_DIR"
echo ""

# Run analysis
python brain_behavior_nki.py \
  --ig_csv "$IG_CSV" \
  --behavioral_file "$CAARS_FILE" \
  --output_dir "$OUTPUT_DIR" \
  --n_components 10

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "================================================================================================"
    echo "✅ NKI-RS TD analysis complete!"
    echo "📊 Results saved to: $OUTPUT_DIR"
    echo "================================================================================================"
else
    echo "================================================================================================"
    echo "❌ NKI-RS TD analysis failed with exit code $EXIT_CODE"
    echo "================================================================================================"
fi

exit $EXIT_CODE

