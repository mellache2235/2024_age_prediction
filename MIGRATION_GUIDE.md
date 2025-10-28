# Migration Guide: Converting Scripts to use plot_styles.py

This guide shows how to update the remaining 8 plotting scripts to use the centralized `plot_styles.py` module for 100% consistent formatting.

---

## ✅ Already Completed (2/10)

- `run_nki_brain_behavior_enhanced.py`
- `run_adhd200_brain_behavior_enhanced.py`

---

## 🔄 Need to Update (8/10)

### Brain-Behavior Scripts (3)
1. `run_cmihbn_brain_behavior_enhanced.py`
2. `run_adhd200_adhd_brain_behavior_enhanced.py`  
3. `run_cmihbn_adhd_brain_behavior_enhanced.py`

### Combined Plotting Scripts (2)
4. `plot_brain_behavior_td_cohorts.py`
5. `plot_brain_behavior_custom_1x3.py`

### Brain Age Scripts (3)
6. `plot_brain_age_td_cohorts.py`
7. `plot_brain_age_adhd_cohorts.py`
8. `plot_brain_age_asd_cohorts.py`

---

## 📋 Step-by-Step Migration Pattern

### For Brain-Behavior Scripts (run_*_brain_behavior_enhanced.py)

#### Step 1: Update imports (top of file)

**Replace this**:
```python
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import matplotlib.backends.backend_pdf as pdf
import seaborn as sns
# ... other imports ...

# Set Arial font
font_path = '/oak/.../arial.ttf'
font_manager.fontManager.addfont(font_path)
# ... font setup ...

from logging_utils import (...)
```

**With this**:
```python
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf_backend
# ... other imports (sklearn, scipy, etc.) ...

# Add to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'utils'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from logging_utils import (...)
from plot_styles import get_dataset_title, setup_arial_font, create_standardized_scatter, DPI, FIGURE_FACECOLOR

# Setup Arial font globally
setup_arial_font()
```

#### Step 2: Find the `create_scatter_plot()` function

Search for the function that creates scatter plots (usually ~50 lines of matplotlib code).

#### Step 3: Replace with centralized version

**Replace entire function with**:
```python
def create_scatter_plot(y_actual, y_pred, rho, p_value, behavioral_name, dataset_name, output_dir):
    """Create scatter plot using centralized styling."""
    # Format p-value
    p_str = "< 0.001" if p_value < 0.001 else f"= {p_value:.3f}"
    
    # Create stats text
    stats_text = f"ρ = {rho:.3f}\np {p_str}"
    
    # Get standardized title
    title = get_dataset_title(dataset_name)
    
    # Create safe filename
    safe_name = behavioral_name.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
    save_path = Path(output_dir) / f'scatter_{safe_name}'
    
    # Use centralized plotting function
    fig, ax = plt.subplots(figsize=(8, 6))
    
    create_standardized_scatter(
        ax, y_actual, y_pred,
        title=title,
        xlabel='Observed Behavioral Score',
        ylabel='Predicted Behavioral Score',
        stats_text=stats_text,
        is_subplot=False
    )
    
    # Save with centralized export (PNG + TIFF + AI)
    plt.tight_layout()
    
    png_path = save_path.with_suffix('.png')
    tiff_path = save_path.with_suffix('.tiff')
    ai_path = save_path.with_suffix('.ai')
    
    plt.savefig(png_path, dpi=DPI, bbox_inches='tight', facecolor=FIGURE_FACECOLOR, edgecolor='none')
    plt.savefig(tiff_path, dpi=DPI, bbox_inches='tight', facecolor=FIGURE_FACECOLOR, edgecolor='none',
               format='tiff', pil_kwargs={'compression': 'tiff_lzw'})
    pdf_backend.FigureCanvas(fig).print_pdf(str(ai_path))
    
    plt.close()
    
    print(f"  ✓ Saved: {png_path.name} + {tiff_path.name} + {ai_path.name}")
```

---

### For Brain Age Scripts (plot_brain_age_*.py)

#### Similar pattern but slightly different stats:

**Replace the scatter plotting section in the loop with**:
```python
# Instead of custom ax.scatter(), ax.plot(), ax.text(), etc.
create_standardized_scatter(
    ax, actual_ages, predicted_ages,
    title=get_dataset_title(dataset_name),
    xlabel='Chronological Age (years)',
    ylabel='Predicted Brain Age (years)',
    stats_text=f"R² = {r_squared:.3f}\nMAE = {mae:.2f} years\nP {p_text}\nN = {len(actual_ages)}",
    is_subplot=True  # Important: True for multi-panel figures
)
```

And update the save section to export PNG + TIFF + AI.

---

## 🎯 Benefits After Migration

✅ **100% consistency** - All plots identical
✅ **Less code** - ~50 lines → ~15 lines per plot
✅ **Easy updates** - Change plot_styles.py once → all plots update
✅ **Triple export** - PNG + TIFF + AI automatic
✅ **Proper titles** - "ADHD-200 TD Subset (NYU)" etc.

---

## 📝 Testing

After updating each script:
```bash
python {script_name}.py
```

Check output:
- Should see: "Saved: plot.png + plot.tiff + plot.ai"
- Verify all 3 files created
- Open PNG - verify styling matches plot_styles.py parameters

---

## 🚨 Common Issues

**"name 'plt' is not defined"**: Add `import matplotlib.pyplot as plt` to imports

**"name 'pdf_backend' is not defined"**: Add `import matplotlib.backends.backend_pdf as pdf_backend`

**Title shows wrong text**: Use `get_dataset_title(dataset_name)` not manual string formatting

---

**Last Updated**: 2024  
**Context**: 700K/1000K tokens used

