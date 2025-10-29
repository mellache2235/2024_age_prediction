# Repository Syncing: Local ↔ Oak

## Repository Setup

This repository exists in two locations:

**Local (Development)**:
```
/Users/hari/Desktop/SCSNL/2024_age_prediction/
```

**Oak (Production/Analysis)**:
```
/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/
```

---

## Syncing Process

When you make changes locally, sync to Oak:

### Option 1: Git Clone/Push (Recommended)
```bash
# Local: commit and push changes
cd /Users/hari/Desktop/SCSNL/2024_age_prediction
git add .
git commit -m "Updated optimization scripts"
git push

# Oak: pull changes
ssh oak
cd /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test
git pull
```

### Option 2: rsync
```bash
rsync -av --exclude='.git' \
  /Users/hari/Desktop/SCSNL/2024_age_prediction/ \
  oak:/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/
```

### Option 3: Manual Copy (Specific Files)
```bash
# Copy optimization scripts
scp scripts/run_*_optimized.py \
    oak:/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/scripts/

scp scripts/optimized_brain_behavior_core.py \
    oak:/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/scripts/
```

---

## What to Sync

### Always Sync:
- All optimization scripts (`run_*_optimized.py`)
- Core module (`optimized_brain_behavior_core.py`)
- Utility scripts (`create_optimization_summary_figure.py`, etc.)
- Updated documentation (README files)

### Don't Need to Sync:
- Test outputs
- Local-only documentation
- Backup files

---

## After Syncing

All scripts work the same on both locations:

**Local**:
```bash
cd /Users/hari/Desktop/SCSNL/2024_age_prediction/scripts
python run_all_cohorts_brain_behavior_optimized.py --cohort stanford_asd
python run_stanford_asd_brain_behavior_optimized.py
python run_nki_brain_behavior_optimized.py
```

**Oak** (after sync):
```bash
ssh oak
cd /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/scripts
python run_all_cohorts_brain_behavior_optimized.py --cohort abide_asd
python run_stanford_asd_brain_behavior_optimized.py
python run_nki_brain_behavior_optimized.py
```

---

## Quick Sync Command

Create an alias for easy syncing:

```bash
# Add to ~/.bashrc or ~/.zshrc
alias sync-to-oak='rsync -av --exclude=".git" --exclude="test_output" \
  /Users/hari/Desktop/SCSNL/2024_age_prediction/ \
  oak:/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/'

# Then just run:
sync-to-oak
```

---

## ✅ Current Status

All optimization scripts are ready in **local workspace**:
- ✅ `run_all_cohorts_brain_behavior_optimized.py` (5 cohorts)
- ✅ `run_stanford_asd_brain_behavior_optimized.py`
- ✅ `run_nki_brain_behavior_optimized.py` ← **FIXED VERSION NOW IN WORKSPACE**
- ✅ `optimized_brain_behavior_core.py`
- ✅ All utility scripts
- ✅ Complete documentation

**Next step**: Sync to Oak using your preferred method (git/rsync/scp)

---

**Once synced, all 7 cohorts will work perfectly on Oak!** 🎉

