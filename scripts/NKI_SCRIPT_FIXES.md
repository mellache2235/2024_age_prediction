# Fixes for NKI Optimized Script on Oak

## Issues Found

### Issue 1: `print_info()` Argument Mismatch
```
TypeError: print_info() missing 1 required positional argument: 'value'
```

**Location**: Line 373 in `run_nki_brain_behavior_optimized.py` on Oak

**Problem**: The Oak script uses an older version of `print_info()` that expects `(key, value)` format, but the code is calling it with a single string.

**Solution**: Replace all `print_info()` calls with the two-argument format.

---

### Issue 2: Subject ID Column Not Found
```
ValueError: No subject ID column found in Conners Parent. 
Available columns: ['Anonymized ID', ...]
```

**Problem**: ID detection doesn't recognize 'Anonymized ID'

---

## Complete Fix

Replace these functions in `/oak/.../run_nki_brain_behavior_optimized.py`:

### Fix 1: Update `standardize_subject_id()` (around line 270-285)

```python
def standardize_subject_id(df, file_name):
    """Standardize subject ID column to 'subject_id'."""
    # Look for ID column (FLEXIBLE MATCHING - includes 'anonymized')
    id_col = None
    for col in df.columns:
        col_lower = col.lower()
        # Check for various ID column patterns
        if any(keyword in col_lower for keyword in ['id', 'subject', 'identifier', 'anonymized']):
            id_col = col
            break
    
    if id_col is None:
        raise ValueError(f"No subject ID column found in {file_name}. Available columns: {list(df.columns)}")
    
    # Rename to 'subject_id'
    if id_col != 'subject_id':
        df = df.rename(columns={id_col: 'subject_id'})
    
    # Convert to string
    df['subject_id'] = df['subject_id'].astype(str)
    
    return df
```

**Key change**: Added `'anonymized'` to the keyword list.

---

### Fix 2: Update `load_ig_scores()` (around line 325-375)

Replace all `print_info()` calls to use two arguments:

```python
def load_ig_scores():
    """Load IG scores from CSV with integrity checks."""
    print_step("Loading IG scores")
    print(f"From {Path(IG_CSV).name}")
    print("-" * 100)
    
    ig_df = pd.read_csv(IG_CSV)
    
    # Integrity check on raw DataFrame
    check_data_integrity(ig_df, "IG DataFrame (raw)")
    
    # Extract subject IDs and IG scores
    subject_ids = ig_df['subject_id'].values
    ig_cols = [col for col in ig_df.columns if col.startswith('ROI_')]
    ig_matrix = ig_df[ig_cols].values
    
    # Integrity check on IG matrix
    check_data_integrity(ig_matrix, "IG matrix", subject_ids)
    
    print_info("IG subjects", len(subject_ids))              # Two arguments
    print_info("IG features (ROIs)", ig_matrix.shape[1])     # Two arguments
    
    return subject_ids, ig_matrix, ig_cols
```

**Key changes**: 
- `print_info(f"IG subjects: {len(subject_ids)}", 0)` → `print_info("IG subjects", len(subject_ids))`
- All `print_info()` calls now use `(key, value)` format

---

### Fix 3: Update `load_nki_behavioral_data()` (around line 267-323)

Replace the entire function with this enhanced version:

```python
def load_nki_behavioral_data():
    """Load and merge all NKI behavioral data files (from enhanced script logic)."""
    print_step("Loading NKI behavioral data from multiple files")
    
    behavioral_dir = Path(BEHAVIORAL_DIR)
    
    # Find all behavioral files
    behavioral_files = []
    for pattern in ['*CAARS*.csv', '*Conners*.csv', '*RBS*.csv']:
        behavioral_files.extend(behavioral_dir.glob(pattern))
    
    if not behavioral_files:
        raise ValueError(f"No behavioral files found in {behavioral_dir}")
    
    print_info("Found behavioral files", len(behavioral_files))
    for f in behavioral_files:
        print(f"  • {f.name}")
    
    # Load and merge all behavioral files
    all_dfs = []
    for bf in behavioral_files:
        df = pd.read_csv(bf)
        
        # Identify subject ID column (FLEXIBLE - includes 'anonymized')
        id_col = None
        for col in df.columns:
            col_lower = col.lower()
            if any(kw in col_lower for kw in ['id', 'subject', 'identifier', 'anonymized']):
                id_col = col
                break
        
        if id_col is None:
            print(f"⚠ WARNING: No subject ID column in {bf.name}, skipping")
            continue
        
        # Standardize to 'subject_id'
        if id_col != 'subject_id':
            df = df.rename(columns={id_col: 'subject_id'})
        
        # Convert subject IDs to string
        df['subject_id'] = df['subject_id'].astype(str)
        
        all_dfs.append(df)
        print_info(f"Loaded from {bf.name}", len(df))
    
    # Merge all dataframes on subject_id
    if not all_dfs:
        raise ValueError("No valid behavioral files found")
    
    merged_df = all_dfs[0]
    for df in all_dfs[1:]:
        merged_df = pd.merge(merged_df, df, on='subject_id', how='outer')
    
    # Remove duplicate columns (ending with _x or _y from merge)
    duplicate_cols = [col for col in merged_df.columns if col.endswith('_x') or col.endswith('_y')]
    if duplicate_cols:
        print_info("Removing duplicate columns", len(duplicate_cols))
        merged_df = merged_df.drop(columns=duplicate_cols)
    
    # Auto-detect behavioral columns
    behavioral_cols = [col for col in merged_df.columns if 
                       'CAARS' in col.upper() or 
                       'CONNERS' in col.upper() or
                       'RBS' in col.upper() or
                       ('TOTAL' in col.upper() and ('INATTENTION' in col.upper() or 'HYPERACTIVITY' in col.upper())) or
                       ('T-SCORE' in col.upper() or 'T_SCORE' in col.upper())]
    
    # Exclude subject_id
    behavioral_cols = [col for col in behavioral_cols if col != 'subject_id']
    
    if not behavioral_cols:
        raise ValueError("No behavioral columns found")
    
    print_info("Total behavioral subjects", len(merged_df))
    print_info("Behavioral measures", len(behavioral_cols))
    print(f"Sample columns: {behavioral_cols[:5]}...")
    
    return merged_df, behavioral_cols
```

**Key changes**:
- All `print_info()` calls use `(key, value)` format
- Added `'anonymized'` to ID keyword detection
- Simplified and matches enhanced script logic

---

## Quick Apply Instructions

```bash
# 1. SSH to Oak
ssh oak

# 2. Navigate to scripts directory
cd /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/scripts

# 3. Backup current file
cp run_nki_brain_behavior_optimized.py run_nki_brain_behavior_optimized.py.backup

# 4. Edit the file
nano run_nki_brain_behavior_optimized.py

# 5. Apply fixes:
#    - In standardize_subject_id(): Add 'anonymized' to keywords
#    - In load_ig_scores(): Change print_info() to two-argument format
#    - In load_nki_behavioral_data(): Replace entire function with version above

# 6. Test
python run_nki_brain_behavior_optimized.py
```

---

## Or Use the Working Enhanced Script Logic

The simplest fix is to copy the data loading functions directly from:
`/oak/.../scripts/run_nki_brain_behavior_enhanced.py`

The enhanced script works perfectly, so just copy its:
1. `load_nki_behavioral_data()` function
2. ID detection logic (lines 100-162 in enhanced script)

---

## Expected Output After Fix

```
[STEP] Loading NKI behavioral data from multiple files
----------------------------------------------------------------------------------------------------
Found behavioral files: 4
  • 8100_CAARS_20191009.csv
  • 8100_Conners_3-P(S)_20191009.csv
  • 8100_Conners_3-SR(S)_20191009.csv
  • 8100_RBS-R_20191009.csv

Loaded from 8100_CAARS_20191009.csv: 245
Loaded from 8100_Conners_3-P(S)_20191009.csv: 187
...

Total behavioral subjects: 352
Behavioral measures: 45
Sample columns: ['CAARS_36', 'CAARS_37', ...]

[STEP] Merging IG and behavioral data
✓ Merged: 245 subjects with both IG and behavioral data

[Then optimization proceeds normally...]
```

---

**Issue**: `print_info()` API mismatch + ID column detection  
**Solution**: Use two-argument format + add 'anonymized' keyword  
**Status**: Fix documented, ready to apply

