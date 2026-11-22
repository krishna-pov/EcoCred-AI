# Project Context: Above-Ground Biomass (AGB) Estimation using GEDI, Sentinel-1/2, and XGBoost

This document summarizes the full state of an ongoing machine learning project for other AI models/collaborators to pick up with full context. It covers the original notebook structure, model performance history, diagnosis of issues, changes already made, and changes planned but not yet implemented.

---

## 1. Project Goal

Predict above-ground biomass (AGBD, in Mg/ha) at a given location using satellite remote sensing features (optical, radar, terrain), trained against GEDI spaceborne LiDAR as ground truth. This is a regression problem solved with XGBoost.

**Study area:** Bounding box over India — `ee.Geometry.BBox(75.0, 18.0, 85.0, 28.0)`
**Time window:** 2022-01-01 to 2023-12-31

---

## 2. Original Notebook Structure (as uploaded: `Carbon_Stock.ipynb`)

### Cell 4 — GEDI ground truth extraction (ORIGINAL, since modified — see Section 4)
- Loads `LARSE/GEDI/GEDI04_A_002_MONTHLY` image collection, filtered to study area and date range
- Quality filter: only `l4_quality_flag == 1`
- Median composite of `agbd` band, clipped to study area
- Stratifies AGBD into 3 classes: 1 (0–50 Mg/ha), 2 (50–100 Mg/ha), 3 (>100 Mg/ha)
- `stratifiedSample()` pulls `numPoints=1000` total across strata, scale=50m, with geometries, tileScale=16

### Cell 5 — Feature extraction (Sentinel-2 optical + Sentinel-1 radar) — INCOMPLETE, PLANNED FOR NEXT UPDATE (see Section 5)
- **Sentinel-2 (optical):** `COPERNICUS/S2_SR_HARMONIZED`, cloud-masked via QA60 bitmask (cloud bit 10, cirrus bit 11), median composite
  - Derived indices: NDVI, EVI, SAVI, MSAVI (formulas already implemented)
  - Selected bands: B2, B3, B4, B8, B11, B12, NDVI, EVI, SAVI, MSAVI
- **Sentinel-1 (radar):** `COPERNICUS/S1_GRD`, filtered for VV+VH polarization, IW mode, median composite, bands VV/VH
  - Cell as originally written was **cut off** — ends mid-statement (`s1`) with no GLCM texture code actually written, even though the final exported CSV (`Biomass_Enhanced_Features_2022_2023.csv`) already contains texture columns like `VV_var` and `VV_contrast`. This means the working texture-extraction code exists somewhere (likely was run and then the cell was edited/truncated afterward) but is not currently visible/reproducible in the notebook as-is.

### Cell 7 onward — Data loading and preprocessing (already run against previously exported CSV)
- Loads `Biomass_Enhanced_Features_2022_2023.csv` into `df`
- Drops GEE housekeeping columns: `system:index`, `agb_class`, `.geo`
- Checks nulls
- Splits `y = df['agbd']`, `X = df.drop(columns=['agbd'])`
- `train_test_split(X, y, test_size=0.2, random_state=42)`

### EDA cells
- Histogram of `y_train` (AGBD distribution) — right-skewed
- Correlation heatmap of `X_train` features
- Scatter plots: `canopy_height` vs AGBD, `MSAVI` vs AGBD, `VV_var` vs AGBD, `NDVI` vs AGBD

### Multicollinearity fix
- Dropped columns from both `X_train` and `X_test`: `B2, B3, B11, NDVI, SAVI, VV, VV_contrast, latitude, longitude`
- Re-plotted correlation heatmap after dropping

### Outlier/anomaly removal
- Removed rows where `y_train > 800` (outliers) — same threshold applied to `y_test`
- Removed anomalies where `canopy_height == 0` AND `agbd > 50` (physically implausible — zero canopy height but high biomass)
- Applied consistently and independently to both train and test sets (each computes its own `valid_indices`)

### Target transformation
- `y_train_transformed = np.log1p(y_train)` — log transform to reduce right-skew before modeling
- Histogram plotted to confirm more normal distribution post-transform

### Modeling — XGBoost with RandomizedSearchCV
- Original param grid included a **typo bug**: `cosample_bytree` instead of `colsample_bytree`. XGBoost silently ignores unrecognized parameters, so this hyperparameter was NEVER actually tuned in the original runs — it stayed at default (1.0) across all trials.
- Grid (as originally written, with typo):
  ```python
  param_grid = {
      'n_estimators' : [50,100,200,300],
      'learning_rate' : [0.01,0.05,0.1,0.2],
      'max_depth' : [3,5,7,10],
      'subsample' : [0.7,0.8,0.9,1.0],
      'cosample_bytree' : [0.7,0.8,0.9,1]   # BUG: should be colsample_bytree
  }
  ```
- `RandomizedSearchCV(estimator=xgb.XGBRegressor(random_state=42), param_distributions=param_grid, n_iter=25, scoring='neg_mean_squared_error', cv=5, n_jobs=-1, random_state=42)`
- Fit on `X_train, y_train_transformed`
- Best estimator extracted (`best_xgb_model`), feature importances pulled
- Evaluation: predictions inverse-transformed with `np.expm1()`, then R², MSE, RMSE, MAE computed against original-scale `y_train`/`y_test`

---

## 3. Model Performance History (chronological)

| Run | n_iter | colsample_bytree tuned? | Train R² | Train RMSE | Train MAE | Test R² | Test RMSE | Test MAE |
|---|---|---|---|---|---|---|---|---|
| Run 1 (original, with typo bug) | 25 | No (typo, stuck at default) | 0.6605 | 41.91 | 25.54 | 0.6338 | 38.66 | 25.70 |
| Run 2 (typo fixed) | 25 | Yes | 0.6269 | 43.94 | 26.92 | 0.6169 | 39.54 | 26.36 |
| Run 3 (typo fixed, n_iter increased) | 75 | Yes | 0.6269 | 43.94 | 26.92 | 0.6169 | 39.54 | 26.36 |

**Key diagnosis:** Train and test R² are close together in all runs (no overfitting) — the model is **underfitting**. This means the ceiling is set by data volume, feature quality, and label noise, NOT by hyperparameter tuning. Fixing the `colsample_bytree` typo was correct to do (for search-space correctness) but caused a slight apparent regression because it expanded the search space (4×4×4×4×4 = 1024 combos vs. previous 256) without proportionally increasing `n_iter`, making the random search cover a sparser fraction of a larger space. Increasing `n_iter` from 25→75 did not meaningfully change results, confirming tuning has hit its ceiling — **further hyperparameter search is not expected to be productive without first improving the underlying data.**

**Conclusion reached:** Stop spending time on hyperparameter tuning for now. Focus on data volume, label quality, and feature engineering — these are expected to have far greater impact on R² than further search.

---

## 4. Changes Already Made (Cell 4 — GEDI extraction)

The user has approved and asked to update Cell 4 with the following changes (this has been written and handed to the user to run, but **results of running it have not yet been reported back**):

```python
import ee
import time

# 1. Study area
study_area = ee.Geometry.BBox(75.0, 18.0, 85.0, 28.0)

# 2. Load GEDI and Clean — stricter quality filtering
gedi = ee.ImageCollection('LARSE/GEDI/GEDI04_A_002_MONTHLY') \
         .filterBounds(study_area) \
         .filterDate('2022-01-01', '2023-12-31')

def filter_quality(img):
    quality_mask = img.select('l4_quality_flag').eq(1)
    degrade_mask = img.select('degrade_flag').eq(0)
    sensitivity_mask = img.select('sensitivity').gte(0.95)
    full_mask = quality_mask.And(degrade_mask).And(sensitivity_mask)
    return img.select('agbd').updateMask(full_mask)

gedi_clean = gedi.map(filter_quality).median().clip(study_area)

# 3. Stratify the Biomass into Classes
# Binning into classes: 1 (0-50 Mg/ha), 2 (50-100 Mg/ha), 3 (>100 Mg/ha)
agb_strata = ee.Image(1) \
    .where(gedi_clean.select('agbd').gt(50), 2) \
    .where(gedi_clean.select('agbd').gt(100), 3) \
    .toInt() \
    .rename('agb_class')

gedi_with_strata = gedi_clean.addBands(agb_strata)

# 4. Stratified Sampling with the Integer Band — larger sample size
samples = gedi_with_strata.stratifiedSample(
    numPoints=6000,
    classBand='agb_class',
    region=study_area,
    scale=50,
    geometries=True,
    tileScale=16
)

print("Cell 1 Complete: Ground truth point collection (with strata) initiated.")
print("Sample count:", samples.size().getInfo())
```

**Rationale for each change:**
1. Added `degrade_flag == 0` filter — removes GEDI shots affected by spacecraft attitude/pointing degradation during acquisition
2. Added `sensitivity >= 0.95` filter — GEDI's own documentation recommends this threshold for reliable biomass retrievals; low-sensitivity shots have significantly higher error, especially over dense canopy where the laser signal is attenuated
3. `numPoints` increased from 1000 → 6000 — original sample size (further shrunk by outlier/anomaly removal downstream) was thin for a model with 15+ correlated features; more GEDI points is expected to be the single biggest lever for improving performance
4. Added a diagnostic print of `samples.size().getInfo()` to verify the stricter filtering doesn't over-filter any one stratum (if a class ends up with very few points, the sensitivity threshold may need to be relaxed to 0.9)

**Status: Written and provided to user. Awaiting user to run this cell in their GEE environment and report back the resulting sample count and whether it proceeds as expected before moving to the next cell.**

---

## 5. Changes Planned But NOT Yet Implemented

### 5a. Cell 5 — Radar feature engineering (Sentinel-1 + PALSAR-2)

This is the next cell to update once Cell 4's new sampling is confirmed working. Planned changes, combining our own diagnosis with the one validated suggestion from a second AI (Gemini) consulted by the user:

**From our own analysis:**
- Complete the truncated Sentinel-1 GLCM texture extraction (the original cell cut off before finishing)
- Add VH-band GLCM texture (currently only VV texture exists in the exported CSV — `VV_var`, `VV_contrast` — but no VH texture)
- Add **VH/VV ratio** and **VH−VV difference** as new derived features — these do not saturate at high biomass the way raw VV/VH backscatter does, directly addressing the known SAR saturation problem (~100–150 Mg/ha) that likely explains a meaningful chunk of current model error, especially since the AGBD histogram shows substantial mass in that range
- Add terrain features from `USGS/SRTMGL1_003`: elevation, slope, aspect (via `ee.Terrain.slope()` / `ee.Terrain.aspect()`) — free, global, and often meaningfully predictive of biomass through correlation with species composition, land-use pressure, and water availability; not yet added to any cell

**From Gemini's suggestions (evaluated for practicality):**
Gemini proposed three sensor upgrades to get past R² 0.80: Airborne LiDAR (ALS), L-band radar (ALOS PALSAR-2), and Hyperspectral imagery. Of these three, only **ALOS PALSAR-2 was judged genuinely actionable** in this project's context (GEE-based, no-cost workflow):
- **Airborne LiDAR (ALS):** Correct in theory (gold-standard ground truth), but requires an actual aircraft survey — not a queryable global dataset. Not accessible for this study area. Rejected as impractical for now.
- **Hyperspectral imagery:** Correct in theory (could help distinguish species/wood density), but no freely available, globally-consistent hyperspectral archive exists on GEE comparable to Sentinel-2's coverage. Sparse/non-systematic datasets like PRISMA or EMIT would require a fundamentally different acquisition strategy. Rejected as impractical for now.
- **L-band Radar (ALOS PALSAR-2): ACCEPTED — this is the one useful, actionable suggestion from Gemini.** Unlike C-band Sentinel-1 (which is dominated by canopy leaf/branch scattering and saturates early), L-band's longer wavelength (~23cm) penetrates the canopy and interacts with larger woody structures — trunks and major branches — giving a truer signal of wood volume at higher biomass levels where Sentinel-1 saturates. This is well-established in the biomass-SAR literature. It is freely available on GEE (`JAXA/ALOS/PALSAR-2/Level2_2/ScanSAR` or the yearly mosaic `JAXA/ALOS/PALSAR/YEARLY/SAR`), requires no special licensing, and is a straightforward drop-in addition to the existing radar processing pipeline in Cell 5.

**Planned Cell 5 will therefore include, in combination:**
1. Existing Sentinel-2 optical processing (NDVI, EVI, SAVI, MSAVI) — unchanged
2. Completed Sentinel-1 VV+VH GLCM texture extraction (fixing the truncation)
3. New VH/VV ratio and VH−VV difference bands
4. New ALOS PALSAR-2 L-band radar bands (to be added to the feature stack — specific band selection and GLCM texture on PALSAR-2, if desired, not yet finalized)
5. New SRTM terrain bands: elevation, slope, aspect

This has been discussed and outlined conversationally but **no code has yet been written or delivered for Cell 5** — this is the next concrete task.

### 5b. Downstream re-work required once new data lands
Once the GEE re-extraction (Cells 4 and 5) produces a new, larger, richer CSV export, the following downstream cells will need to be re-run/re-checked (not yet done, flagged for awareness):
- Multicollinearity check (Cell 24 equivalent) — new features (VH_VV_ratio, elevation, slope, PALSAR-2 bands) need a fresh correlation heatmap; the previous drop list (`B2, B3, B11, NDVI, SAVI, VV, VV_contrast, latitude, longitude`) may not fully apply to the expanded feature set
- Outlier/anomaly thresholds (`y_train <= 800`, `canopy_height == 0 & agbd > 50`) — should be re-validated against the new, larger dataset's distribution (re-plot histogram)
- Hyperparameter search — once real data-driven gains are captured, re-run `RandomizedSearchCV` (with the typo already fixed) at `n_iter=75-100`, since search stability improves with a larger, more representative training set

### 5c. Other ideas discussed but deferred (not yet started, no code written)
- **Two-stage / stratified modeling approach:** Since the data is already explicitly binned into 3 AGB classes, consider training separate regressors per stratum, or a classifier-then-regressor cascade — motivated by the idea that low-biomass regions may be more optical/NDVI-driven while high-biomass regions are more structure/SAR-driven, i.e., the underlying relationship may not be well-served by a single global regressor across the full range.
- **Seasonal compositing instead of one blended 2022–2023 median:** Try separate wet-season and dry-season composites for both S1 and S2, added as additional feature sets, rather than relying on a single blended median composite that may obscure useful seasonal vegetation structure/moisture signal.
- **Spatial leakage check on train/test split:** The current split is a plain random 80/20 split (`train_test_split` with `random_state=42`). Worth checking whether spatially adjacent GEDI points fall into both train and test sets, which could either inflate apparent performance (if very close points leak across the split) or understate true generalization (if points are naturally clustered). No spatial-blocking or buffered split has been implemented yet.

---

## 6. Realistic Expectation Set for the User

It was communicated to the user that:
- Published GEDI-based biomass studies at similar resolution and using similar feature sets (optical + SAR + terrain) typically land in the R² 0.6–0.75 range, even with well-engineered features.
- Reaching R² > 0.80 is an achievable stretch goal but not guaranteed with GEDI-derived ground truth alone — it more commonly requires either denser/better ground truth (e.g., field plots or ALS-derived reference biomass, not available here) or a substantially larger training set than currently available.
- The single highest-expected-ROI next steps, in order, are: (1) more GEDI points with stricter quality filtering [already actioned in Cell 4], (2) SAR saturation mitigation via VH/VV ratio + PALSAR-2 L-band [planned for Cell 5], (3) terrain features [planned for Cell 5], (4) re-tuning hyperparameters only after the above data improvements are in place.

---

## 7. Immediate Next Step

Waiting on the user to run the updated Cell 4 (GEDI extraction with stricter quality filters and 6000-point stratified sample) in their Google Earth Engine environment and report the resulting sample count / any issues, before proceeding to write the updated Cell 5 (Sentinel-1 texture completion + VH/VV ratio + ALOS PALSAR-2 + SRTM terrain features).
