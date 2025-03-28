import os, datetime, math, requests
import pandas as pd, numpy as np
from shapely.geometry import LineString
import folium
import xarray as xr
from metpy.units import units
import metpy.calc as mpcalc
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader


# ————— CARGA DE CONFIGURACIÓN —————
with open("config.yaml") as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config["credentials"],
    config["cookie"]["name"],
    config["cookie"]["key"],
    config["cookie"]["expiry_days"]
)

                      # ← Detiene ejecución


try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo
LOCAL_TZ = ZoneInfo("America/Mexico_City")

# -------------------------- Funciones comunes --------------------------

def geocode(place_name):
    url = f"https://nominatim.openstreetmap.org/search?format=json&q={place_name}"
    r = requests.get(url, headers={"User-Agent": "MiApp/1.0"})
    data = r.json()
    if not data:
        raise Exception(f"No se encontró {place_name}")
    return float(data[0]["lat"]), float(data[0]["lon"])

def get_route_osrm(origin, destination):
    url = (
        f"http://router.project-osrm.org/route/v1/driving/"
        f"{origin['lon']},{origin['lat']};{destination['lon']},{destination['lat']}?overview=full&geometries=geojson"
    )
    r = requests.get(url)
    if r.status_code != 200:
        raise Exception("Error OSRM")
    return r.json()["routes"][0]["geometry"]["coordinates"]

def segment_route(coords, start_time, speed_kmh, km_step=10):
    line = LineString(coords)
    length_km = line.length * 111
    n = math.ceil(length_km / km_step)
    segments = []
    for i in range(n + 1):
        frac = i / n
        pt = line.interpolate(frac, normalized=True)
        hours = (length_km * frac) / speed_kmh
        segments.append({
            "segment_id": i,
            "lat": pt.y,
            "lon": pt.x,
            "time": start_time + datetime.timedelta(hours=hours)
        })
    return segments

def interpolate(ds, var, lat, lon, t):
    # Selecciona el valor más cercano para acelerar el proceso
    val = ds[var].sel(time=t, lat=lat, lon=lon, method="nearest").compute().values
    return float(val) if val is not None else np.nan

def forecast_point(seg, ds):
    t = seg["time"]
    t2 = interpolate(ds, "T2", seg["lat"], seg["lon"], t) - 273.15
    rain = interpolate(ds, "RAINNC", seg["lat"], seg["lon"], t)
    u10 = interpolate(ds, "U10", seg["lat"], seg["lon"], t)
    v10 = interpolate(ds, "V10", seg["lat"], seg["lon"], t)
    ws = (mpcalc.wind_speed(u10 * units("m/s"), v10 * units("m/s")).to("km/h").magnitude
          if not np.isnan(u10 + v10) else np.nan)
    risk = "low"
    if rain > 5 or ws > 60:
        risk = "medium"
    if rain > 15 or ws > 80:
        risk = "high"
    return {
        "segment_id": seg["segment_id"],
        "time_utc": t.isoformat(),
        "latitude": seg["lat"],
        "longitude": seg["lon"],
        "temp_c": round(t2, 1),
        "rain_mm_h": round(rain, 2),
        "wind_km_h": round(ws, 1),
        "risk_level": risk
    }

def route_forecast_real(origin, destination, start_time, speed, ds):
    coords = get_route_osrm(origin, destination)
    segments = segment_route(coords, start_time, speed)
    with ThreadPoolExecutor(max_workers=16) as executor:  # Ajusta max_workers según el entorno
        forecast = list(executor.map(lambda seg: forecast_point(seg, ds), segments))
    return forecast, coords

def generar_mapa(coords, forecast, origin, destination):
    mid = [(origin["lat"] + destination["lat"]) / 2,
           (origin["lon"] + destination["lon"]) / 2]
    m = folium.Map(location=mid, zoom_start=7)
    folium.PolyLine([(y, x) for x, y in coords], color="blue", weight=4).add_to(m)
    for seg in forecast:
        folium.Marker(
            [seg["latitude"], seg["longitude"]],
            popup=folium.Popup(f"{seg}", max_width=300),
            icon=folium.Icon(color={"low": "green", "medium": "orange", "high": "red"}[seg["risk_level"]])
        ).add_to(m)
    m.save("ruta_map.html")

# -------------------------- Cargar dataset desde Zarr --------------------------
@st.cache_data(show_spinner=False)
def load_dataset_zarr():
    BASE_DIR = Path(__file__).parent.resolve()
    zarr_path = BASE_DIR / "wrf_actual.zarr"
    # Carga el dataset sin chunking
    ds = xr.open_zarr(zarr_path)
    # Aplica el rechunking después de la carga
    ds = ds.chunk({"time": 1, "lat": 50, "lon": 50})
    return ds


ds = load_dataset_zarr()

# -------------------------- Función principal (Streamlit) --------------------------
def main_streamlit():
    st.title("Pronóstico de Ruta con WRF")
    origen = st.text_input("Origen", "Ciudad de México", key="origen")
    destino = st.text_input("Destino", "Veracruz", key="destino")
    hora_local_str = st.text_input("Hora Local (YYYY-MM-DD HH:MM)", 
                               datetime.datetime.now(tz=LOCAL_TZ).strftime("%Y-%m-%d %H:%M"),
                               key="hora")
    velocidad = st.number_input("Velocidad km/h", 80, key="vel")
    
    if st.button("Obtener Pronóstico", key="btn"):
        try:
            user_local = datetime.datetime.strptime(hora_local, "%Y-%m-%d %H:%M")
        except ValueError:
            st.error("Formato incorrecto — usa YYYY-MM-DD HH:MM")
            return
        
        # Convertir hora local a UTC y luego seleccionar el tiempo más cercano en ds
        start_utc = user_local.replace(tzinfo=LOCAL_TZ).astimezone(datetime.timezone.utc).replace(tzinfo=None)
        nearest = pd.to_datetime(ds.time.sel(time=start_utc, method="nearest").values)
        start = nearest.to_pydatetime()
        
        lat_o, lon_o = geocode(origen)
        lat_d, lon_d = geocode(destino)
        
        forecast, coords = route_forecast_real(
            {"lat": lat_o, "lon": lon_o},
            {"lat": lat_d, "lon": lon_d},
            start, velocidad, ds
        )
        df = pd.DataFrame(forecast)
        # Convertir time_utc a hora local (como texto) sin problemas de tz
        df["time_local"] = df["time_utc"].apply(
            lambda s: datetime.datetime.fromisoformat(s)
                        .replace(tzinfo=datetime.timezone.utc)
                        .astimezone(LOCAL_TZ)
                        .strftime("%Y-%m-%d %H:%M:%S %Z")
        )
        
        st.subheader("Pronóstico de la Ruta")
        st.write(df[["segment_id", "time_local", "temp_c", "rain_mm_h", "wind_km_h", "risk_level"]])
        
        generar_mapa(coords, forecast, {"lat": lat_o, "lon": lon_o}, {"lat": lat_d, "lon": lon_d})
        with open("ruta_map.html") as f:
            st.components.v1.html(f.read(), height=600)

# ————— LOGIN —————
authenticator.login(location="main")

status = st.session_state.get("authentication_status")

if status:
    name = st.session_state["name"]
    st.write(f"✅ Bienvenido, **{name}**")
    authenticator.logout("Cerrar sesión", "main")
    main_streamlit()               # ← Solo aquí ejecuto tu app

elif status is False:
    st.error("❌ Usuario o contraseña incorrectos")
    st.stop()                      # ← Detiene ejecución

else:
    st.info("🔒 Ingresa tus credenciales para acceder")
    st.stop()
