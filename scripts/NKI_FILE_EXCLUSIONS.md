# NKI Behavioral File Exclusions

## Files Included

The NKI optimization script now loads **ONLY**:
- ✅ CAARS (Conners Adult ADHD Rating Scale)
- ✅ Conners 3 Parent Rating (Conners 3-P)
- ✅ Conners 3 Self-Report (Conners 3-SR)

## Files Excluded

The following are **explicitly excluded**:
- ❌ YRBS (Youth Risk Behavior Survey) - Not relevant for brain-behavior
- ❌ DKEFS Proverbs - Not relevant
- ❌ RBS-R (Repetitive Behavior Scale) - Not relevant for ADHD focus

## Why Exclude?

1. **Focus on ADHD-relevant measures**: CAARS and Conners are gold standard for ADHD
2. **Reduces noise**: Fewer irrelevant variables
3. **Faster processing**: Fewer columns to analyze
4. **Better results**: Focus optimization on meaningful measures

## What You'll See

### Before (loading all):
```
Found 7 behavioral files:
  • 8100_CAARS_20191009.csv
  • 8100_Conners_3-P(S)_20191009.csv
  • 8100_Conners_3-SR(S)_20191009.csv
  • 8100_RBS-R_20191009.csv
  • 8100_YRBS-MS_20191009.csv
  • 8100_YRBS-HS_20191009.csv
  • 8100_DKEFS_PROVERBS_20191009.csv

Behavioral measures: 150+
```

### After (CAARS/Conners only):
```
Found 3 behavioral files:
  • 8100_CAARS_20191009.csv
  • 8100_Conners_3-P(S)_20191009.csv
  • 8100_Conners_3-SR(S)_20191009.csv

Behavioral measures: 40-60 (focused on ADHD/attention)
```

## Behavioral Measures Now Included

From CAARS and Conners 3, you'll get measures like:
- Inattention/Memory Problems
- Hyperactivity/Restlessness
- Impulsivity/Emotional Lability
- Problems with Self-Concept
- DSM-5 Inattentive Symptoms
- DSM-5 Hyperactive-Impulsive Symptoms
- ADHD Index
- Executive Functioning
- Learning Problems
- Aggression
- Peer Relations

**All relevant for ADHD and attention-related brain-behavior correlations!**

## Implementation

Updated in: `run_nki_brain_behavior_optimized.py`

```python
# Find ONLY CAARS and Conners files
behavioral_files = []
for pattern in ['*CAARS*.csv', '*Conners*.csv']:
    behavioral_files.extend(BEHAVIORAL_DIR.glob(pattern))

# Explicitly exclude unwanted files
excluded_patterns = ['YRBS', 'DKEFS', 'RBS', 'Proverbs', 'proverbs']
behavioral_files = [f for f in behavioral_files 
                   if not any(excl in f.name for excl in excluded_patterns)]
```

---

**Status**: ✅ Fixed  
**Effect**: Focuses on ADHD-relevant measures only  
**Runtime**: Faster (fewer measures to optimize)

