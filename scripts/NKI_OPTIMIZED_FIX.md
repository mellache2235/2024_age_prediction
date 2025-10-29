# Fix for NKI Optimized Script

## Error
```
ValueError: No subject ID column found in Conners Parent. 
Available columns: ['Anonymized ID', 'Subject Type', ...]
```

## Problem
The `standardize_subject_id()` function in `/oak/.../run_nki_brain_behavior_optimized.py` is not finding 'Anonymized ID'.

## Solution

Replace the `standardize_subject_id()` function (around line 270-285) with this:

```python
def standardize_subject_id(df, file_name):
    """Standardize subject ID column to 'subject_id'."""
    # Look for ID column (flexible matching)
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

## Or Use the Enhanced NKI Logic

Replace the entire `load_nki_behavioral_data()` function (around line 267-323) with:

```python
def load_nki_behavioral_data():
    """Load and merge all NKI behavioral data files (from enhanced script)."""
    print_step("Loading NKI behavioral data from multiple files")
    
    behavioral_dir = Path(BEHAVIORAL_DIR)
    
    # Find all behavioral files
    behavioral_files = []
    for pattern in ['*CAARS*.csv', '*Conners*.csv', '*RBS*.csv']:
        behavioral_files.extend(behavioral_dir.glob(pattern))
    
    if not behavioral_files:
        raise ValueError(f"No behavioral files found in {behavioral_dir}")
    
    print_info(f"Found {len(behavioral_files)} behavioral files:")
    for f in behavioral_files:
        print_info(f"  • {f.name}")
    
    # Load and merge all behavioral files
    all_dfs = []
    for bf in behavioral_files:
        df = pd.read_csv(bf)
        
        # Identify subject ID column (FLEXIBLE MATCHING)
        id_col = None
        for col in df.columns:
            col_lower = col.lower()
            if 'id' in col_lower or 'subject' in col_lower or 'anonymized' in col_lower:
                id_col = col
                break
        
        if id_col is None:
            print_warning(f"No subject ID column in {bf.name}, skipping")
            continue
        
        # Standardize to 'subject_id'
        if id_col != 'subject_id':
            df = df.rename(columns={id_col: 'subject_id'})
        
        # Convert subject IDs to string
        df['subject_id'] = df['subject_id'].astype(str)
        
        all_dfs.append(df)
        print_info(f"  Loaded {len(df)} subjects from {bf.name}")
    
    # Merge all dataframes on subject_id
    if not all_dfs:
        raise ValueError("No valid behavioral files found")
    
    merged_df = all_dfs[0]
    for df in all_dfs[1:]:
        merged_df = pd.merge(merged_df, df, on='subject_id', how='outer')
    
    # Remove duplicate columns
    duplicate_cols = [col for col in merged_df.columns if col.endswith('_x') or col.endswith('_y')]
    if duplicate_cols:
        print_info(f"Removing {len(duplicate_cols)} duplicate columns")
        merged_df = merged_df.drop(columns=duplicate_cols)
    
    # Auto-detect behavioral columns (CAARS, Conners, RBS)
    behavioral_cols = [col for col in merged_df.columns if 
                       'CAARS' in col.upper() or 
                       'CONNERS' in col.upper() or
                       'RBS' in col.upper() or
                       ('TOTAL' in col.upper() and ('INATTENTION' in col.upper() or 'HYPERACTIVITY' in col.upper())) or
                       ('T-SCORE' in col.upper() or 'T_SCORE' in col.upper())]
    
    # Exclude subject_id if it got included
    behavioral_cols = [col for col in behavioral_cols if col != 'subject_id']
    
    if not behavioral_cols:
        raise ValueError("No behavioral columns found")
    
    print_info(f"Total behavioral subjects: {len(merged_df)}")
    print_info(f"Behavioral measures: {len(behavioral_cols)}")
    print_info(f"Sample columns: {behavioral_cols[:5]}...")
    
    return merged_df, behavioral_cols
```

## Key Changes

1. **Flexible ID matching**: Checks for 'id', 'subject', 'identifier', 'anonymized' in column name
2. **Matches enhanced script exactly**: Same logic as working NKI enhanced script
3. **Better error messages**: Shows what columns are available

## Apply the Fix

```bash
# 1. Edit the file on Oak
nano /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/scripts/run_nki_brain_behavior_optimized.py

# 2. Find the standardize_subject_id function (around line 270-285)

# 3. Add 'anonymized' to the keyword list:
if any(keyword in col_lower for keyword in ['id', 'subject', 'identifier', 'anonymized']):

# 4. Save and exit

# 5. Test
python run_nki_brain_behavior_optimized.py
```

---

**Issue**: ID column detection too strict  
**Solution**: Use flexible matching like enhanced script  
**Status**: Fix documented, ready to apply

