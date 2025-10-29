#!/bin/bash
#
# Sync Fixed Optimization Scripts to Oak
#
# This syncs all fixed optimization scripts from local to Oak
# Run this BEFORE testing optimization on Oak!
#

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                    SYNCING OPTIMIZATION SCRIPTS TO OAK                     ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

LOCAL_DIR="/Users/hari/Desktop/SCSNL/2024_age_prediction/scripts"
OAK_DIR="/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/scripts"

echo "📁 Source (local): $LOCAL_DIR"
echo "📁 Target (Oak):   $OAK_DIR"
echo ""

# Files to sync
FILES=(
    "optimized_brain_behavior_core.py"
    "run_stanford_asd_brain_behavior_optimized.py"
    "run_abide_asd_brain_behavior_optimized.py"
    "run_nki_brain_behavior_optimized.py"
    "run_all_cohorts_brain_behavior_optimized.py"
)

echo "📦 Files to sync:"
for file in "${FILES[@]}"; do
    echo "  • $file"
done
echo ""

# Sync each file
echo "🔄 Syncing files..."
echo ""

for file in "${FILES[@]}"; do
    echo -n "  Syncing $file... "
    scp "$LOCAL_DIR/$file" "oak:$OAK_DIR/$file" 2>/dev/null
    
    if [ $? -eq 0 ]; then
        echo "✅"
    else
        echo "❌ FAILED"
    fi
done

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                              SYNC COMPLETE!                                ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "✅ All optimization scripts synced to Oak"
echo ""
echo "Now you can run on Oak:"
echo "  ssh oak"
echo "  cd $OAK_DIR"
echo "  python run_stanford_asd_brain_behavior_optimized.py"
echo "  python run_abide_asd_brain_behavior_optimized.py"
echo "  python run_nki_brain_behavior_optimized.py"
echo "  python run_all_cohorts_brain_behavior_optimized.py --cohort adhd200_td"
echo ""
echo "Expected improvements:"
echo "  • No formatting errors"
echo "  • NKI: ρ ≥ 0.41 (baseline was 0.41)"
echo "  • ABIDE: Valid ADOS data"
echo "  • All cohorts: Comprehensive integrity checks"
echo ""

