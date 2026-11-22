
import numpy as np
import streamlit as st
import folium
import rasterio
from rasterio.mask import mask
from shapely.geometry import shape, mapping, box
from pyproj import Geod
from folium.plugins import Draw
from streamlit_folium import st_folium

BIOMASS_FILE = "biomass_map.tif"
MODEL_ERROR = 41.0
MAX_VALID_AGB = 400.0

st.set_page_config(
    page_title="Carbon Credit Estimator",
    layout="wide"
)

st.title("Carbon Credit Estimator")
st.caption(
    "Draw your land on the satellite map to estimate "
    "a range of plausible carbon credits."
)

# ---------- Assumptions ----------
with st.sidebar:
    st.header("Assumptions")
    st.caption(
        "Placeholder defaults. Replace with values from "
        "the methodology you follow."
    )

    root_shoot = st.slider(
        "Root-to-shoot ratio (below-ground biomass)",
        0.10, 0.60, 0.24, 0.01
    )

    growth_low, growth_high = st.slider(
        "Yearly carbon gain, as % of stored biomass",
        0.5, 10.0, (2.0, 5.0), 0.5
    )

    buffer_pct = st.slider(
        "Buffer / safety deduction (%)",
        0, 40, 15
    )

# ---------- Read biomass map bounds ----------
try:
    with rasterio.open(BIOMASS_FILE) as src:
        west, south, east, north = src.bounds

except Exception as e:
    st.error(
        f"Could not open {BIOMASS_FILE}. "
        "Place the biomass map next to app.py. "
        f"Details: {e}"
    )
    st.stop()

st.caption(
    f"Demo region: {south:.2f} to {north:.2f} N, "
    f"{west:.2f} to {east:.2f} E. "
    "Draw your land fully inside the blue boundary."
)

# ---------- Satellite map ----------
center_lat = (south + north) / 2
center_lon = (west + east) / 2

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=12,
    tiles=None,
    control_scale=True
)

# Esri satellite imagery basemap
folium.TileLayer(
    tiles=(
        "https://server.arcgisonline.com/ArcGIS/rest/services/"
        "World_Imagery/MapServer/tile/{z}/{y}/{x}"
    ),
    attr=(
        "Tiles &copy; Esri — Source: Esri, Maxar, "
        "Earthstar Geographics, and the GIS User Community"
    ),
    name="Satellite Imagery",
    overlay=False,
    control=True,
    show=True
).add_to(m)

# Keep the demo boundary
folium.Rectangle(
    bounds=[
        [south, west],
        [north, east]
    ],
    color="blue",
    weight=3,
    fill=False,
    tooltip="Demo region covered by this app"
).add_to(m)

# Keep polygon and rectangle drawing tools
Draw(
    export=False,
    draw_options={
        "polygon": True,
        "rectangle": True,
        "circle": False,
        "marker": False,
        "polyline": False,
        "circlemarker": False
    },
    edit_options={
        "edit": True,
        "remove": True
    }
).add_to(m)

# Add layer control
folium.LayerControl(collapsed=True).add_to(m)

# Fit map to the biomass coverage
m.fit_bounds([
    [south, west],
    [north, east]
])

out = st_folium(
    m,
    height=550,
    width=None,
    key="satellite_land_map"
)

# ---------- Get drawn geometry ----------
drawing = out.get("last_active_drawing") if out else None

if drawing is None:
    st.info(
        "Use the polygon or rectangle drawing tool "
        "on the satellite map to outline your land."
    )
    st.stop()

geom = shape(drawing["geometry"])

# ---------- Validate drawn area ----------
demo_region = box(west, south, east, north)

if not geom.is_valid or geom.is_empty:
    st.error("The drawn geometry is invalid. Please redraw your land.")
    st.stop()

if not geom.within(demo_region):
    st.error(
        "Your area is not fully inside the blue demo region. "
        "Please redraw it inside the boundary."
    )
    st.stop()

# ---------- Calculate area ----------
area_m2 = abs(
    Geod(ellps="WGS84").geometry_area_perimeter(geom)[0]
)

area_ha = area_m2 / 10_000

if area_ha <= 0:
    st.error("The selected area must be greater than zero.")
    st.stop()

# ---------- Read biomass from GeoTIFF ----------
try:
    with rasterio.open(BIOMASS_FILE) as src:
        data, _ = mask(
            src,
            [mapping(geom)],
            crop=True,
            nodata=np.nan,
            filled=True
        )

        values = data[0].astype("float64")

except ValueError:
    st.error(
        "Could not extract biomass data for this area. "
        "Try drawing a different region."
    )
    st.stop()

except Exception as e:
    st.error(f"Error reading biomass data: {e}")
    st.stop()

# Remove invalid and negative biomass values
values = values[
    np.isfinite(values) & (values >= 0)
]

if values.size == 0:
    st.error(
        "No valid biomass data exists inside this area. "
        "Try another location within the demo region."
    )
    st.stop()

agb_mean = float(values.mean())

agb_low = max(agb_mean - MODEL_ERROR, 0.0)
agb_high = agb_mean + MODEL_ERROR

# ---------- Carbon calculations ----------
def to_co2e_per_ha(agb):
    total_biomass = agb * (1 + root_shoot)
    carbon = total_biomass * 0.47
    return carbon * 44 / 12


stock_low = to_co2e_per_ha(agb_low) * area_ha
stock_high = to_co2e_per_ha(agb_high) * area_ha

keep = 1 - buffer_pct / 100

credits_low = (
    stock_low * growth_low / 100 * keep
)

credits_high = (
    stock_high * growth_high / 100 * keep
)

# ---------- Display results ----------
st.subheader("Results")

c1, c2, c3 = st.columns(3)

c1.metric(
    "Area",
    f"{area_ha:,.2f} ha"
)

c2.metric(
    "Estimated biomass",
    f"{agb_mean:,.1f} Mg/ha"
)

c3.metric(
    "CO₂ stock (range)",
    f"{stock_low:,.1f} to {stock_high:,.1f} t"
)

st.success(
    f"Estimated carbon credits per year: "
    f"**{credits_low:,.1f} to {credits_high:,.1f}** "
    f"(1 credit = 1 t CO₂e)"
)

if agb_mean > MAX_VALID_AGB * 0.9:
    st.warning(
        "Biomass is near the upper limit of the model's "
        "validated range (400 Mg/ha). Treat the estimate "
        "with caution."
    )

st.caption(
    "This is a demonstration estimate, not a verified "
    "carbon-credit amount. Real credits require an eligible "
    "methodology, baseline, additionality assessment, and "
    "appropriate monitoring and verification. The yearly "
    "growth and buffer values are assumptions that you can adjust."
)
