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
from st_aggrid import AgGrid, GridOptionsBuilder  # Componente de autocompletado

# --- Configurar zona horaria ---
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo
LOCAL_TZ = ZoneInfo("America/Mexico_City")

##########################################
# FUNCIONES NUEVAS
##########################################

# 1. Filtro: Verifica que las coordenadas estén en México
def dentro_de_mexico(lat, lon):
    # Aproximadamente: latitud entre 14 y 33, longitud entre -118 y -86
    return 14 <= lat <= 33 and -118 <= lon <= -86

# 2. Buscar localidades: Consulta Nominatim limitado a México y devuelve un DataFrame
def buscar_localidades_df(query):
    url = f"https://nominatim.openstreetmap.org/search?format=json&q={query}&countrycodes=mx"
    r = requests.get(url, headers={"User-Agent": "MiApp/1.0"})
    if r.status_code == 200:
        resultados = r.json()
        if resultados:
            df = pd.DataFrame(resultados)
            # Nos aseguramos de tener las columnas necesarias
            if "display_name" in df.columns and "lat" in df.columns and "lon" in df.columns:
                df = df[["display_name", "lat", "lon"]]
                df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
                df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
                return df
    return pd.DataFrame()

# 3. Etiquetado global de ruta según riesgo
def etiqueta_ruta(forecast):
    total = len(forecast)
    high = sum(1 for p in forecast if p["risk_level"] == "high")
    medium = sum(1 for p in forecast if p["risk_level"] == "medium")
    if total == 0:
        return "Sin datos"
    if high / total > 0.3:
        return "Ruta insegura"
    elif medium / total > 0.3:
        return "Ruta poco segura"
    else:
        return "Ruta muy segura"

##########################################
# FUNCIONES EXISTENTES
##########################################

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
    with ThreadPoolExecutor(max_workers=16) as executor:
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

@st.cache_data(show_spinner=False)
def load_dataset_zarr():
    BASE_DIR = Path(__file__).parent.resolve()
    zarr_path = BASE_DIR / "wrf_actual.zarr"
    ds = xr.open_zarr(zarr_path)
    ds = ds.chunk({"time": 1, "lat": 50, "lon": 50})
    return ds

ds = load_dataset_zarr()

##########################################
# CONFIGURACIÓN DE LOGIN
##########################################
with open("config.yaml") as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config["credentials"],
    config["cookie"]["name"],
    config["cookie"]["key"],
    config["cookie"]["expiry_days"]
)

##########################################
# FUNCIÓN PRINCIPAL STREAMLIT
##########################################
def main_streamlit():
    st.title("Pronóstico de Ruta con WRF")
    
    st.write("### Ingrese las localidades")
    
    # Autocompletar para Origen
    origen_query = st.text_input("Localidad de Origen", "Ciudad de México", key="origen_query")
    if origen_query:
        df_origen = buscar_localidades_df(origen_query)
        if not df_origen.empty:
            gb_origen = GridOptionsBuilder.from_dataframe(df_origen)
            gb_origen.configure_default_column(filter=True, sortable=True)
            gridOptions_origen = gb_origen.build()
            st.markdown("#### Seleccione una opción de Origen")
            grid_response_origen = AgGrid(df_origen, gridOptions=gridOptions_origen, update_mode="SELECTION_CHANGED", height=200, fit_columns_on_grid_load=True)
            selected_origen = grid_response_origen.get("selected_rows")
            if selected_origen:
                origen_lat = float(selected_origen[0]["lat"])
                origen_lon = float(selected_origen[0]["lon"])
            else:
                origen_lat, origen_lon = geocode("Ciudad de México")
        else:
            origen_lat, origen_lon = geocode("Ciudad de México")
    else:
        origen_lat, origen_lon = geocode("Ciudad de México")
    
    # Autocompletar para Destino
    destino_query = st.text_input("Localidad de Destino", "Veracruz", key="destino_query")
    if destino_query:
        df_destino = buscar_localidades_df(destino_query)
        if not df_destino.empty:
            gb_destino = GridOptionsBuilder.from_dataframe(df_destino)
            gb_destino.configure_default_column(filter=True, sortable=True)
            gridOptions_destino = gb_destino.build()
            st.markdown("#### Seleccione una opción de Destino")
            grid_response_destino = AgGrid(df_destino, gridOptions=gridOptions_destino, update_mode="SELECTION_CHANGED", height=200, fit_columns_on_grid_load=True)
            selected_destino = grid_response_destino.get("selected_rows")
            if selected_destino:
                destino_lat = float(selected_destino[0]["lat"])
                destino_lon = float(selected_destino[0]["lon"])
            else:
                destino_lat, destino_lon = geocode("Veracruz")
        else:
            destino_lat, destino_lon = geocode("Veracruz")
    else:
        destino_lat, destino_lon = geocode("Veracruz")
    
    # Filtro: Verificar que ambas localizaciones estén en México
    if not (dentro_de_mexico(origen_lat, origen_lon) and dentro_de_mexico(destino_lat, destino_lon)):
        st.error("La ruta debe estar dentro de México.")
        return
    
    # No se muestran las coordenadas, solo se usan internamente.
    
    # Hora y velocidad
    utc_now = datetime.datetime.now(datetime.timezone.utc)
    default_time = utc_now.astimezone(LOCAL_TZ).strftime("%Y-%m-%d %H:%M")
    st.write("UTC ahora:", utc_now.strftime("%Y-%m-%d %H:%M:%S %Z"))
    st.write("Valor por defecto local:", default_time)
    hora_local = st.text_input("Hora Local (YYYY-MM-DD HH:MM)", default_time, key="hora")
    velocidad = st.number_input("Velocidad km/h", 80, key="vel")
    
    if st.button("Obtener Pronóstico", key="btn"):
        try:
            user_local = datetime.datetime.strptime(hora_local, "%Y-%m-%d %H:%M")
            user_local = user_local.replace(tzinfo=LOCAL_TZ)
        except ValueError:
            st.error("Formato incorrecto — usa YYYY-MM-DD HH:MM")
            return
        
        # Convertir a UTC naive para buscar en ds.time
        start_utc = user_local.astimezone(datetime.timezone.utc).replace(tzinfo=None)
        nearest = pd.to_datetime(ds.time.sel(time=start_utc, method="nearest").values)
        start = nearest.to_pydatetime()
        
        # Obtener pronóstico
        forecast, coords = route_forecast_real(
            {"lat": origen_lat, "lon": origen_lon},
            {"lat": destino_lat, "lon": destino_lon},
            start, velocidad, ds
        )
        df = pd.DataFrame(forecast)
        df["time_local"] = df["time_utc"].apply(
            lambda s: datetime.datetime.fromisoformat(s)
                        .replace(tzinfo=datetime.timezone.utc)
                        .astimezone(LOCAL_TZ)
                        .strftime("%Y-%m-%d %H:%M:%S %Z")
        )
        
        st.subheader("Pronóstico de la Ruta")
        st.dataframe(df[["segment_id", "time_local", "temp_c", "rain_mm_h", "wind_km_h", "risk_level"]])
        
        # Etiquetado global de la ruta
        ruta_etiqueta = etiqueta_ruta(forecast)
        st.write("**Evaluación general de la ruta:**", ruta_etiqueta)
        
        generar_mapa(coords, forecast, {"lat": origen_lat, "lon": origen_lon}, {"lat": destino_lat, "lon": destino_lon})
        with open("ruta_map.html") as f:
            st.components.v1.html(f.read(), height=600)

##########################################
# CONFIGURACIÓN DE LOGIN
##########################################
authenticator.login(location="main")
status = st.session_state.get("authentication_status")

if status:
    name = st.session_state["name"]
    st.write(f"✅ Bienvenido, **{name}**")
    authenticator.logout("Cerrar sesión", "main")
    main_streamlit()  # Se ejecuta la app solo si el login es exitoso.
elif status is False:
    st.error("❌ Usuario o contraseña incorrectos")
    st.stop()
else:
    st.info("🔒 Ingresa tus credenciales para acceder")
    st.stop()

