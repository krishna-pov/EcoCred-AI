"""
STEP 2 (run once, no Earth Engine needed): turn the exported features into biomass_map.tif.

Needs in the same folder:
  nitr_features.tif        (downloaded from Google Drive after export_features.py)
  biomass_model.joblib     (saved from the notebook)
  feature_columns.json     (saved from the notebook)
"""
import json
import joblib
import numpy as np
import rasterio

FEATURES_TIF = "nitr_features.tif"
MODEL_FILE = "biomass_model.joblib"
COLUMNS_FILE = "feature_columns.json"
OUT_FILE = "biomass_map.tif"

NODATA_IN = -9999.0
NODATA_OUT = -9999.0
MAX_AGB = 400.0                      # model was trained and tested only up to 400 Mg/ha
NON_VEGETATED = {50, 70, 80, 100}    # built-up, snow/ice, water, moss: set to 0 biomass
                                     # (the model saw almost no examples of these)

model = joblib.load(MODEL_FILE)
columns = json.load(open(COLUMNS_FILE))

with rasterio.open(FEATURES_TIF) as src:
    band_names = list(src.descriptions)
    print("Bands in file:", band_names)
    if "LandCover" not in band_names:
        raise SystemExit("LandCover band not found in the feature file.")

    needed = [c for c in columns if not c.startswith("lc_")]
    missing = [c for c in needed if c not in band_names]
    if missing:
        raise SystemExit(f"These model features are missing from the file: {missing}")

    profile = src.profile.copy()
    profile.update(count=1, dtype="float32", nodata=NODATA_OUT,
                   compress="lzw", tiled=True, blockxsize=256, blockysize=256)

    out = np.full((src.height, src.width), NODATA_OUT, dtype="float32")
    CHUNK = 256  # rows at a time, to keep memory low

    for r0 in range(0, src.height, CHUNK):
        r1 = min(r0 + CHUNK, src.height)
        win = rasterio.windows.Window(0, r0, src.width, r1 - r0)
        arr = src.read(window=win).astype("float64")          # (bands, rows, cols)
        n_rows = r1 - r0

        def band(name):
            return arr[band_names.index(name)].reshape(-1)

        lc = np.rint(band("LandCover")).astype(int)

        # Build the model input in exactly the training column order
        cols = []
        for c in columns:
            if c.startswith("lc_"):
                code = int(c.split("_")[1])
                cols.append((lc == code).astype("float64"))
            else:
                cols.append(band(c))
        X = np.column_stack(cols)

        # A pixel is valid only if every continuous feature has data
        cont_idx = [i for i, c in enumerate(columns) if not c.startswith("lc_")]
        valid = np.all(X[:, cont_idx] != NODATA_IN, axis=1) & np.all(np.isfinite(X), axis=1)

        pred = np.full(X.shape[0], NODATA_OUT, dtype="float64")
        if valid.any():
            p = np.expm1(model.predict(X[valid]))              # back from log scale to Mg/ha
            pred[valid] = np.clip(p, 0, MAX_AGB)

        # Non-vegetated land cover -> zero biomass (where land cover is known)
        non_veg = np.isin(lc, list(NON_VEGETATED)) & (band("LandCover") != NODATA_IN)
        pred[non_veg] = 0.0

        out[r0:r1, :] = pred.reshape(n_rows, src.width).astype("float32")
        print(f"  rows {r0}-{r1} of {src.height} done")

with rasterio.open(OUT_FILE, "w", **profile) as dst:
    dst.write(out, 1)

good = out[out != NODATA_OUT]
print(f"\nSaved {OUT_FILE}")
print(f"Valid pixels: {good.size:,} of {out.size:,}")
if good.size:
    print(f"Biomass (Mg/ha): min {good.min():.1f}, mean {good.mean():.1f}, max {good.max():.1f}")
