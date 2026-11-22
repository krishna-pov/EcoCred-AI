"""
STEP 1 (run once): export the model's input features for a small demo region.

Uses Earth Engine one time. After the file is in your Google Drive, the app never needs Earth Engine again.
Feature definitions are copied from Cell 5 of the notebook so they match what the model was trained on.
"""
import time
import ee

# ---------------- EDIT THESE ----------------
PROJECT = "carbonestimationversion2"   # your Earth Engine project id
# West, South, East, North in degrees. Must be INSIDE the training area (75-85 E, 18-28 N).
# Keep it about 0.5 x 0.5 degrees or smaller so the file stays small.
REGION_BBOX = [84.85, 22.20, 84.95, 22.30]   # NIT Rourkela campus with a generous margin (about 11 km x 11 km)
# --------------------------------------------

ee.Initialize(project=PROJECT)
region = ee.Geometry.Rectangle(REGION_BBOX)

# ---- Sentinel-2 ----
def mask_s2_clouds(image):
    qa = image.select('QA60')
    mask = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
    return image.updateMask(mask)

def add_indices(image):
    evi = image.expression(
        '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 10000))',
        {'NIR': image.select('B8'), 'RED': image.select('B4'), 'BLUE': image.select('B2')}).rename('EVI')
    msavi = image.expression(
        '(2 * NIR + 1 - sqrt(pow((2 * NIR + 1), 2) - 8 * (NIR - RED))) / 2',
        {'NIR': image.select('B8'), 'RED': image.select('B4')}).rename('MSAVI')
    return image.addBands([evi, msavi])

s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
      .filterBounds(region).filterDate('2022-01-01', '2023-12-31')
      .map(mask_s2_clouds).map(add_indices).median()
      .select(['B4', 'B8', 'B12', 'EVI', 'MSAVI']))

# ---- Sentinel-1 ----
s1_col = (ee.ImageCollection('COPERNICUS/S1_GRD')
          .filterBounds(region).filterDate('2022-01-01', '2023-12-31')
          .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
          .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
          .filter(ee.Filter.eq('instrumentMode', 'IW'))
          .median().select(['VV', 'VH']))

vv_int = s1_col.select('VV').multiply(100).add(3000).toInt()
vh_int = s1_col.select('VH').multiply(100).add(3000).toInt()
glcm_vv = vv_int.glcmTexture(size=3).select(['VV_var', 'VV_ent'])
glcm_vh = vh_int.glcmTexture(size=3).select(['VH_contrast', 'VH_ent'])
vh_vv_ratio = s1_col.select('VH').divide(s1_col.select('VV')).rename('VH_VV_ratio')
vh_vv_diff = s1_col.select('VH').subtract(s1_col.select('VV')).rename('VH_VV_diff')
s1 = s1_col.select(['VH']).addBands([glcm_vv, glcm_vh, vh_vv_ratio, vh_vv_diff])

# ---- PALSAR-2 (converted from raw values to dB) ----
palsar_dn = (ee.ImageCollection('JAXA/ALOS/PALSAR/YEARLY/SAR_EPOCH')
             .filterBounds(region).filterDate('2022-01-01', '2023-12-31')
             .mosaic().select(['HH', 'HV']))
palsar = palsar_dn.pow(2).log10().multiply(10).subtract(83.0).rename(['PALSAR_HH', 'PALSAR_HV'])

# ---- Terrain, canopy height, land cover ----
srtm = ee.Image('USGS/SRTMGL1_003')
elevation = srtm.select('elevation')
slope = ee.Terrain.slope(elevation).rename('slope')
aspect = ee.Terrain.aspect(elevation).rename('aspect')
canopy_height = ee.Image("users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1").rename('canopy_height')
worldcover = ee.ImageCollection("ESA/WorldCover/v200").first().select('Map').rename('LandCover')

stack = (s2.addBands(s1).addBands(palsar).addBands(elevation)
         .addBands(slope).addBands(aspect).addBands(canopy_height).addBands(worldcover))

# Masked pixels are written as -9999 so the next step can recognise them
stack = stack.unmask(-9999).toFloat()

task = ee.batch.Export.image.toDrive(
    image=stack,
    description='nitr_features',
    folder='EarthEngine_ML_Project',
    fileNamePrefix='nitr_features',
    region=region,
    scale=30,
    maxPixels=1e10,
    fileFormat='GeoTIFF',
)
task.start()
print("Export started. Waiting...")

status = task.status()
while status['state'] in ('READY', 'RUNNING'):
    print(f"  {status['state']}... checking again in 20 seconds")
    time.sleep(20)
    status = task.status()

if status['state'] == 'COMPLETED':
    print("Done. Download nitr_features.tif from the 'EarthEngine_ML_Project' folder in Google Drive.")
else:
    print("Export failed:", status.get('error_message'))
