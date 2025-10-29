# CMI-HBN ADHD Cohort Requires Diagnosis File

## Issue

The universal optimized script fails for CMI-HBN ADHD:
```
ADHD subjects (label='adhd'): 0
```

## Root Cause

**CMI-HBN uses a different ADHD identification method** than other cohorts:

### Other Cohorts (Simple):
- Filter by `label` field in PKLZ
- TD = label='td' or label=0
- ADHD = label='adhd' or label=1

### CMI-HBN (Complex):
- **Requires separate diagnosis CSV file**
- Must load diagnosis file with clinical diagnoses
- Filter by `DX_01_Sub == 'Attention-Deficit/Hyperactivity Disorder'`
- Extract ADHD subject IDs from diagnosis file
- Then filter imaging data by those IDs

---

## From Enhanced Script

```python
# CMI-HBN ADHD enhanced script uses:
DIAGNOSIS_CSV = "/oak/stanford/groups/menon/deriveddata/public/cmihbn/phenotype/9994_ConsensusDx_20190108.csv"

# Steps:
1. Load diagnosis CSV
2. Filter for DX_01_Sub == 'Attention-Deficit/Hyperactivity Disorder'
3. Get ADHD subject IDs
4. Load PKLZ files
5. Filter PKLZ for those ADHD IDs
6. Merge with C3SR
```

---

## Solution

### Option 1: Add Diagnosis File to Universal Script Config

Update cohort config to include diagnosis file path.

### Option 2: Create Dedicated CMI-HBN Scripts (RECOMMENDED)

Like we did for Stanford/ABIDE/NKI, create:
- `run_cmihbn_adhd_brain_behavior_optimized.py`
- `run_cmihbn_td_brain_behavior_optimized.py`

Using EXACT logic from enhanced scripts.

---

## Recommendation

**For now**: Use universal script for TD cohorts only (ADHD200, maybe CMI-HBN TD if label works)

**Create dedicated scripts** for cohorts with complex requirements:
- ✅ Stanford ASD (done)
- ✅ ABIDE ASD (done)
- ✅ NKI (done)
- ⏳ CMI-HBN ADHD (needs diagnosis file)
- ⏳ CMI-HBN TD (may need similar handling)

---

**Status**: Universal script can't handle CMI-HBN ADHD without diagnosis file  
**Workaround**: Create dedicated script using enhanced logic  
**Priority**: Medium (4 other cohorts working)

