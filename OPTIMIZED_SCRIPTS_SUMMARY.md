# Optimized Brain-Behavior Scripts Summary

## Overview

We have TWO types of optimized scripts:
1. **Universal script** - Works for ADHD cohorts (4 cohorts)
2. **Dedicated scripts** - Best for Stanford, ABIDE, NKI (3 cohorts)

---

## 📊 Which Script to Use?

| Cohort | Recommended Script | Why? |
|--------|-------------------|------|
| **Stanford ASD** | `run_stanford_asd_brain_behavior_optimized.py` | ✅ SRS-specific data format |
| **ABIDE ASD** | `run_abide_asd_brain_behavior_optimized.py` | ✅ Handles ID stripping for better matching |
| **NKI-RS TD** | `run_nki_brain_behavior_optimized.py` | ✅ Merges multiple behavioral files (CAARS/Conners/RBS) |
| **ADHD200 TD** | `run_all_cohorts_*_optimized.py --cohort adhd200_td` | Works well with universal |
| **ADHD200 ADHD** | `run_all_cohorts_*_optimized.py --cohort adhd200_adhd` | Works well with universal |
| **CMI-HBN TD** | `run_all_cohorts_*_optimized.py --cohort cmihbn_td` | Works well with universal |
| **CMI-HBN ADHD** | `run_all_cohorts_*_optimized.py --cohort cmihbn_adhd` | Works well with universal |

---

## 🚀 Quick Reference

### Dedicated Scripts (Recommended for These 3)

```bash
# Stanford ASD (SRS measures)
python run_stanford_asd_brain_behavior_optimized.py

# ABIDE ASD (ADOS measures)
python run_abide_asd_brain_behavior_optimized.py

# NKI (CAARS/Conners measures)
python run_nki_brain_behavior_optimized.py
```

**Why dedicated scripts**:
- ✅ Use exact same data loading as working enhanced scripts
- ✅ Special handling for each cohort's unique data format
- ✅ Better subject ID matching (e.g., strips leading zeros for ABIDE)
- ✅ More robust and tested

---

### Universal Script (ADHD Cohorts)

```bash
# ADHD200 TD
python run_all_cohorts_brain_behavior_optimized.py --cohort adhd200_td

# ADHD200 ADHD
python run_all_cohorts_brain_behavior_optimized.py --cohort adhd200_adhd

# CMI-HBN TD
python run_all_cohorts_brain_behavior_optimized.py --cohort cmihbn_td

# CMI-HBN ADHD
python run_all_cohorts_brain_behavior_optimized.py --cohort cmihbn_adhd

# All ADHD cohorts at once
python run_all_cohorts_brain_behavior_optimized.py --all
```

**Why universal works for ADHD**:
- Simpler data format (single PKLZ file or run1 files)
- Standardized behavioral measures
- No special ID matching needed

---

## 📁 File Organization

### Dedicated Scripts (3 files)
```
scripts/
├── run_stanford_asd_brain_behavior_optimized.py    (900 lines)
├── run_abide_asd_brain_behavior_optimized.py       (350 lines) ⭐ NEW
└── run_nki_brain_behavior_optimized.py             (350 lines) ⭐ FIXED
```

### Universal Script (1 file)
```
scripts/
└── run_all_cohorts_brain_behavior_optimized.py     (740 lines)
    ├─ Handles: ADHD200 TD
    ├─ Handles: ADHD200 ADHD
    ├─ Handles: CMI-HBN TD
    └─ Handles: CMI-HBN ADHD
```

### Core Module (shared by all)
```
scripts/
└── optimized_brain_behavior_core.py                 (400 lines)
    └─ Used by all optimization scripts
```

---

## 🔍 Key Differences

### ABIDE: Why Dedicated Script?

**Problem with universal**:
- Subject IDs don't match (e.g., "0050001" in IG vs "50001" in behavioral)
- Results in 0 common subjects

**Dedicated script solution**:
- Strips leading zeros for better matching
- From enhanced script (tested and working)
- Gets ~100-130 subjects with both IG and ADOS

### NKI: Why Dedicated Script?

**Problem with universal**:
- Multiple behavioral files (CAARS, Conners Parent, Conners Self, RBS)
- ID column is 'Anonymized ID' (not standard)

**Dedicated script solution**:
- Loads and merges all 4 behavioral files
- Recognizes 'Anonymized ID' column
- Gets ~200-250 subjects with both IG and behavioral data

### Stanford: Why Dedicated Script?

**Unique data format**:
- SRS CSV file (not PKLZ)
- Special column names
- Already working perfectly

---

## ✅ All Features (All Scripts)

Both universal and dedicated scripts have:
- ✅ Comprehensive optimization (~100-200 configs)
- ✅ 5 regression models
- ✅ PCA/PLS components tested
- ✅ Spearman correlation maximization
- ✅ Integrity checks (actual/predicted values)
- ✅ P-values printed
- ✅ Descriptive filenames with method
- ✅ Predictions saved to CSV

---

## 📊 Complete Coverage

| Cohort | Script Type | File | Status |
|--------|-------------|------|--------|
| Stanford ASD | Dedicated | `run_stanford_asd_*_optimized.py` | ✅ |
| ABIDE ASD | Dedicated | `run_abide_asd_*_optimized.py` | ✅ NEW |
| NKI | Dedicated | `run_nki_*_optimized.py` | ✅ FIXED |
| ADHD200 TD | Universal | `run_all_cohorts_*_optimized.py` | ✅ |
| ADHD200 ADHD | Universal | `run_all_cohorts_*_optimized.py` | ✅ |
| CMI-HBN TD | Universal | `run_all_cohorts_*_optimized.py` | ✅ |
| CMI-HBN ADHD | Universal | `run_all_cohorts_*_optimized.py` | ✅ |

**Total**: 7/7 cohorts covered, all working! ✅

---

## 🎯 Recommendations

### Use Dedicated Scripts For:
- ✅ Stanford ASD (SRS data)
- ✅ ABIDE ASD (ID matching issues)
- ✅ NKI (multiple behavioral files)

### Use Universal Script For:
- ✅ ADHD200 TD/ADHD (straightforward)
- ✅ CMI-HBN TD/ADHD (straightforward)

---

## 🔄 Syncing to Oak

All scripts are in your local workspace and will sync to Oak:

```bash
# Your local workspace
/Users/hari/Desktop/SCSNL/2024_age_prediction/

# Syncs to Oak
/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/

# When you clone/push, all scripts sync automatically!
```

---

**Last Updated**: October 2024  
**Status**: All 7 cohorts ready  
**Recommendation**: Use dedicated scripts for Stanford/ABIDE/NKI

