# Network Aggregation Strategy - Dimensionality Reduction via Neuroscience

## 🧠 Concept

Instead of using 246 individual ROI features, **aggregate them to Yeo network-level features** (7 or 17 networks). This leverages neuroscientific knowledge to create a compact, interpretable feature set.

## 🎯 The Transformation

```
246 ROIs (individual brain regions)
    ↓ Aggregate by Yeo network membership
7-17 Network Features (functional brain systems)
```

**Example Networks:**
- Visual Network
- Somatomotor Network  
- Dorsal Attention Network
- Salience/Ventral Attention Network
- Limbic Network
- Frontoparietal Control Network
- Default Mode Network

## 📊 Why This is Powerful

### For Small Sample Sizes (N < 100):

**Before (ROI-level):**
```
N = 84 subjects
P = 246 ROIs
Ratio = 0.34:1  ❌ Terrible!
Result: Severe overfitting, model collapse
```

**After (Network-level):**
```
N = 84 subjects  
P = 7 networks
Ratio = 12:1  ✅ Excellent!
Result: Stable models, interpretable results
```

### Benefits:

1. **Better Statistical Power**
   - 12:1 subject-to-feature ratio
   - Can use simple models without overfitting
   - More reliable estimates

2. **Noise Reduction**
   - Averaging ROIs within networks reduces measurement noise
   - More stable features
   - Better generalization

3. **Biological Interpretability**
   - "Default Mode Network predicts hyperactivity" 
   - Much more interpretable than "ROI #143 predicts behavior"
   - Can map to known brain-behavior literature

4. **Computational Efficiency**
   - 7 features vs 246 → Much faster
   - Can test more model configurations
   - Enables larger alpha grids

## 🔬 Aggregation Methods

### 1. **Mean** (Default)
```python
# For each network: average IG scores across ROIs
network_score = mean(ROI_IG_scores within network)
```

**Pros:** Standard, interpretable
**Cons:** Positive/negative IGs may cancel out

### 2. **Abs-Mean** 
```python
# For each network: average absolute IG scores
network_score = mean(|ROI_IG_scores| within network)
```

**Pros:** Preserves IG magnitude, no cancellation
**Cons:** Loses sign information

**Both methods are tested**, and the best one is selected via CV!

## 🎯 When Network Aggregation Wins

### Most Likely:
- ✅ **Small N** (N < 100): Excellent ratio improvement
- ✅ **Distributed effects**: Signal spread across many ROIs in a network
- ✅ **High noise**: Individual ROIs are noisy
- ✅ **Network-level hypotheses**: E.g., "DMN relates to social function"

### Less Likely:
- ❌ **Large N** (N > 200): Can handle ROI-level complexity
- ❌ **Focal effects**: If signal is in 2-3 specific ROIs, TopK-IG better
- ❌ **Very weak signals**: Aggregation won't create signal from noise

## 📈 Integration

### Automatically Active:

The strategy is now included in `optimize_comprehensive()` in `optimized_brain_behavior_core.py`.

**All scripts using the core module get it automatically:**
- ✅ `run_all_cohorts_brain_behavior_optimized.py`
- ✅ `run_nki_brain_behavior_optimized.py`
- ✅ `run_abide_asd_brain_behavior_optimized.py`

**Requires manual addition:**
- ⚠️ `run_stanford_asd_brain_behavior_optimized.py` (uses local optimize_comprehensive)

## 🎨 Example Output

### When Network Aggregation Wins:

```
====================================================================================================
BEST PERFORMANCES (Sorted by Spearman ρ)
====================================================================================================
Measure              N_Subjects  Final_Spearman  P_Display  Best_Strategy    Best_Model
Hyperactivity              84        0.387        < 0.001   Network-mean     Ridge
Inattention                81        0.312        0.0041    Network-abs_mean Linear
Impulsivity                84        0.298        0.0067    Network-mean     Lasso
```

### In Optimization Results CSV:

```csv
strategy,model,alpha,n_features,feature_selection,mean_cv_spearman
Network-mean,Ridge,0.1,7,YeoNetworks,0.387
Network-abs_mean,Linear,None,7,YeoNetworks,0.312
TopK-IG,Ridge,0.1,8,MeanAbsValue,0.284
PLS,PLS,15,None,None,0.267
```

### What This Tells You:

**Strategy**: Network-mean
- 246 ROIs aggregated to **7 networks** using **mean**
- **Ridge regression** with α=0.1
- Achieved ρ = 0.387

**Interpretation:**
> "Functional network-level IG scores (averaged within Yeo networks) predict hyperactivity better than individual ROIs. This suggests distributed network effects rather than focal regions."

## 🔍 Technical Details

### Network Mapping:
```python
# Loads from Yeo atlas CSV
YEO_ATLAS_PATH = ".../subregion_func_network_Yeo_updated_yz.csv"

# Creates mapping: ROI index (0-245) → Network ID
network_map = {
    0: 1,    # ROI 0 belongs to Network 1 (Visual)
    1: 1,    # ROI 1 belongs to Network 1
    12: 3,   # ROI 12 belongs to Network 3 (Somatomotor)
    ...
}
```

### Aggregation Process:
```python
For each subject:
    For each network:
        network_score = mean([IG_ROI_j for j in network])

Result: (N_subjects × N_networks) matrix
```

### Models Tested:
- **Linear**: No regularization (best for N:P = 12:1)
- **Ridge**: L2 regularization (alpha grid)
- **Lasso**: L1 regularization (can select subset of networks)

## 🆚 Comparison to Other Strategies

| Strategy | N=84 Ratio | Interpretability | Stability | Speed |
|----------|------------|------------------|-----------|-------|
| PCA (20 comp) | 4.2:1 | ⚠️ Low | ⚠️ OK | Fast |
| PLS (16 comp) | 5.25:1 | ⚠️ Low | ⚠️ Unstable | Fast |
| TopK-IG (8) | 10.5:1 | ✅ Medium | ✅ Good | Fast |
| **Network (7)** | **12:1** | ✅ **High** | ✅ **Best** | **Very Fast** |

## 📊 Expected Performance Gains

### For Small N (like NKI N=81, CMI-HBN N=84):

**Current best (TopK-IG with 8 ROIs):**
```
Spearman ρ: 0.25-0.35
Stability: Good
Interpretability: "8 most important ROIs"
```

**With Network Aggregation:**
```
Spearman ρ: 0.30-0.40 (expected +0.05 improvement)
Stability: Excellent
Interpretability: "Default Mode Network predicts behavior"
```

### Why the Improvement?

1. **Noise Averaging**: Random noise in individual ROIs cancels out
2. **Signal Preservation**: Coherent network signal preserved
3. **Better Generalization**: 12:1 ratio prevents overfitting
4. **Biological Prior**: Networks are the real functional units

## 🚀 Usage

Just run your optimized scripts - no changes needed!

```bash
python run_nki_brain_behavior_optimized.py
python run_all_cohorts_brain_behavior_optimized.py --cohort cmihbn_td
python run_abide_asd_brain_behavior_optimized.py
```

The optimization will automatically:
- Load Yeo atlas mapping
- Test network aggregation alongside other strategies
- Pick the best performer via CV
- Report if network-level features won

## 🔍 Diagnostic Output

When running, you'll see which networks are most important:

```python
# If Network aggregation wins, the feature_selection will be:
Best_Strategy: Network-mean
Feature_Selection: YeoNetworks
N_Features: 7  # Number of networks
Model: Ridge(α=0.1)
```

## 🎁 Bonus: Network-Level Insights

After analysis, you can examine which networks are most predictive:

```python
# From the trained model, extract network coefficients
if best_params['strategy'].startswith('Network'):
    model = best_model.named_steps['regressor']
    network_coefs = model.coef_
    
    # Networks ranked by importance:
    1. Default Mode Network (β=0.45)
    2. Salience Network (β=0.32)
    3. Frontoparietal Network (β=-0.28)
    ...
```

## ⚠️ Limitations

1. **Requires Yeo Atlas File**
   - If file not found, strategy silently skipped
   - No error - just won't be tested

2. **Network Granularity Trade-off**
   - 7 networks: Better for small N, coarser resolution
   - 17 networks: More detailed, but needs larger N

3. **Assumes Network Organization**
   - IG patterns may not align with Yeo networks
   - If signal is cross-network, may dilute it

4. **Aggregation Method Matters**
   - Mean: Can have cancellation if ROIs have opposite signs
   - Abs-mean: Better for IG, but loses directional info
   - Both are tested to find best

## 🔮 Future Enhancements

Potential improvements (not yet implemented):

1. **Weighted aggregation**: Weight ROIs by their IG magnitude
2. **Hierarchical models**: Network-level + within-network detail
3. **Network × Network interactions**: Test cross-network patterns
4. **Subnetwork analysis**: Split large networks (e.g., DMN A/B/C)

## 📚 References

**Yeo 7-Network Parcellation:**
- Yeo et al. (2011). Journal of Neurophysiology.
- Widely used functional brain network parcellation

**Yeo 17-Network Parcellation:**
- Refined version with more granular networks
- Better for larger sample sizes

---

**Status**: ✅ Production Ready
**Added**: 2024
**Author**: Brain-Behavior Optimization Team

