import os
import datetime
import math
import requests
import pandas as pd
import numpy as np
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

# --- Zona horaria ---
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo
LOCAL_TZ = ZoneInfo("America/Mexico_City")

##########################################
# FUNCIONES BÁSICAS
##########################################

def dentro_de_mexico(lat, lon):
    return 14 <= lat <= 33 and -118 <= lon <= -86

@st.cache_data(ttl=3600)
def buscar_localidades_df(query):
    url = (
        "https://nominatim.openstreetmap.org/search"
        f"?format=json&q={query}&countrycodes=mx"
    )
    r = requests.get(url, headers={"User-Agent":"MiApp/1.0"})
    if r.status_code == 200:
        resultados = r.json()
        if resultados:
            df = pd.DataFrame(resultados)
            if {"display_name","lat","lon"}.issubset(df.columns):
                df = df[["display_name","lat","lon"]]
                df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
                df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
                return df
    return pd.DataFrame()

def etiqueta_ruta(forecast):
    total = len(forecast)
    high = sum(1 for p in forecast if p["risk_level"]=="high")
    medium = sum(1 for p in forecast if p["risk_level"]=="medium")
    if total==0: return "Sin datos"
    if high/total > 0.3: return "Ruta insegura"
    if medium/total > 0.3: return "Ruta poco segura"
    return "Ruta muy segura"

def geocode(place_name):
    url = f"https://nominatim.openstreetmap.org/search?format=json&q={place_name}"
    r = requests.get(url, headers={"User-Agent":"MiApp/1.0"})
    data = r.json()
    if not data: raise Exception(f"No se encontró {place_name}")
    return float(data[0]["lat"]), float(data[0]["lon"])

def get_alternative_routes(origin, destination, n_alts=3):
    url = (
        "http://router.project-osrm.org/route/v1/driving/"
        f"{origin['lon']},{origin['lat']};"
        f"{destination['lon']},{destination['lat']}"
        f"?overview=full&geometries=geojson&alternatives={n_alts}"
    )
    r = requests.get(url)
    if r.status_code!=200:
        raise Exception("Error OSRM al solicitar rutas alternativas")
    data = r.json()
    return [rt["geometry"]["coordinates"] for rt in data["routes"]]

def segment_route(coords, start_time, speed_kmh, km_step=10):
    line = LineString(coords)
    length_km = line.length * 111
    n = math.ceil(length_km/km_step)
    segs=[]
    for i in range(n+1):
        frac=i/n
        pt=line.interpolate(frac, normalized=True)
        hours=(length_km*frac)/speed_kmh
        segs.append({
            "segment_id":i,
            "lat":pt.y,
            "lon":pt.x,
            "time":start_time+datetime.timedelta(hours=hours)
        })
    return segs

def interpolate(ds, var, lat, lon, t):
    val = ds[var].sel(time=t, lat=lat, lon=lon, method="nearest").compute().values
    return float(val) if val is not None else np.nan

def forecast_point(seg, ds):
    t2_c = interpolate(ds, "T2", seg["lat"], seg["lon"], seg["time"]) - 273.15
    rain = interpolate(ds, "RAINNC", seg["lat"], seg["lon"], seg["time"])
    u10 = interpolate(ds, "U10", seg["lat"], seg["lon"], seg["time"])
    v10 = interpolate(ds, "V10", seg["lat"], seg["lon"], seg["time"])
    ws_kmh = (mpcalc.wind_speed(u10*units("m/s"), v10*units("m/s"))
              .to("km/h").magnitude if not np.isnan(u10+v10) else np.nan)

    # Wind chill
    if (t2_c<=10) and (ws_kmh>4.8):
        temp_f = (t2_c*units.degC).to(units.degF)
        ws_mph = (ws_kmh*units("km/h")).to(units("mph"))
        wc_f = mpcalc.wind_chill(temp_f, ws_mph)
        wc_c = wc_f.to(units.degC).magnitude
    else:
        wc_c = t2_c

    risk="low"
    if rain>5 or ws_kmh>60: risk="medium"
    if rain>15 or ws_kmh>80: risk="high"

    return {
        "segment_id": seg["segment_id"],
        "time_utc": seg["time"].isoformat(),
        "latitude": seg["lat"],
        "longitude": seg["lon"],
        "temp_c": round(t2_c,1),
        "wind_chill_c": round(wc_c,1),
        "rain_mm_h": round(rain,2),
        "wind_km_h": round(ws_kmh,1),
        "risk_level": risk
    }

def compare_alternative_routes(origin, destination, start_time, speed_kmh, ds, n_alts=3):
    all_coords = get_alternative_routes(origin, destination, n_alts)
    results=[]
    for coords in all_coords:
        segs = segment_route(coords, start_time, speed_kmh)
        with ThreadPoolExecutor(max_workers=8) as ex:
            forecast = list(ex.map(lambda s: forecast_point(s, ds), segs))
        results.append((coords, forecast))
    return results

def summarize_route(forecast):
    tot=len(forecast)
    h=sum(1 for s in forecast if s["risk_level"]=="high")
    m=sum(1 for s in forecast if s["risk_level"]=="medium")
    t0=datetime.datetime.fromisoformat(forecast[0]["time_utc"])
    t1=datetime.datetime.fromisoformat(forecast[-1]["time_utc"])
    eta=int((t1-t0).total_seconds()/60)
    return {
        "Segmentos": tot,
        "% Alto riesgo": f"{100*h/tot:.1f}%",
        "% Medio riesgo": f"{100*m/tot:.1f}%",
        "ETA (min)": eta
    }

def generar_mapa(coords, forecast, origin, destination):
    mid=[(origin["lat"]+destination["lat"])/2, (origin["lon"]+destination["lon"])/2]
    m=folium.Map(location=mid, zoom_start=6)
    colores=["blue","green","orange","purple","red"]
    for i,(c,_f) in enumerate(forecast):
        folium.PolyLine([(lat,lon) for lon,lat in c],
                        color=colores[i], weight=4).add_to(m)
    m.save("comparison_map.html")

@st.cache_data(show_spinner=False)
def load_dataset_zarr():
    base=Path(__file__).parent
    ds=xr.open_zarr(base/"wrf_actual.zarr")
    return ds.chunk({"time":1,"lat":50,"lon":50})

ds=load_dataset_zarr()

##########################################
# LOGIN
##########################################
with open("config.yaml") as f:
    cfg=yaml.load(f,Loader=SafeLoader)
authenticator=stauth.Authenticate(
    cfg["credentials"], cfg["cookie"]["name"],
    cfg["cookie"]["key"], cfg["cookie"]["expiry_days"]
)

##########################################
# APP
##########################################
def main_streamlit():
    st.title("Comparativa de Rutas Alternativas")
    st.write("### Origen y Destino")

    # Inicializar cachés
    if "last_q_ori" not in st.session_state:
        st.session_state["last_q_ori"]=""
        st.session_state["df_ori_cache"]=pd.DataFrame()
    if "last_q_des" not in st.session_state:
        st.session_state["last_q_des"]=""
        st.session_state["df_des_cache"]=pd.DataFrame()

    def autocomplete_point(key, label, default):
        q = st.text_input(label, value=default, key=f"{key}_qry")
        df = pd.DataFrame()
        if q and len(q)>=3:
            if q!=st.session_state[f"last_q_{key}"]:
                st.session_state[f"df_{key}_cache"]=buscar_localidades_df(q)
                st.session_state[f"last_q_{key}"]=q
            df=st.session_state[f"df_{key}_cache"]
        if not df.empty:
            opts=df["display_name"].tolist()
            sel=st.selectbox(f"Selecciona {label}", opts, key=f"{key}_sel")
            row=df[df["display_name"]==sel].iloc[0]
            return row["lat"], row["lon"]
        return geocode(default)

    ori_lat,ori_lon = autocomplete_point("ori","Origen","Ciudad de México")
    des_lat,des_lon = autocomplete_point("des","Destino","Veracruz")

    if not (dentro_de_mexico(ori_lat,ori_lon) and dentro_de_mexico(des_lat,des_lon)):
        st.error("Origen/Destino fuera de México"); return

    utc_now=datetime.datetime.now(datetime.timezone.utc)
    default_time=utc_now.astimezone(LOCAL_TZ).strftime("%Y-%m-%d %H:%M")
    hora_local=st.text_input("Hora Local (YYYY-MM-DD HH:MM)", default_time, key="hora")
    velocidad=st.number_input("Velocidad km/h",80,key="vel")

    if st.button("Comparar rutas alternativas"):
        try:
            ul=datetime.datetime.strptime(hora_local,"%Y-%m-%d %H:%M")
            ul=ul.replace(tzinfo=LOCAL_TZ)
        except ValueError:
            st.error("Formato hora incorrecto"); return
        start_utc=ul.astimezone(datetime.timezone.utc).replace(tzinfo=None)
        nearest=pd.to_datetime(ds.time.sel(time=start_utc,method="nearest").values)
        start=nearest.to_pydatetime()

        n_alts=st.slider("Rutas alternativas",1,5,3)
        results=compare_alternative_routes(
            {"lat":ori_lat,"lon":ori_lon},
            {"lat":des_lat,"lon":des_lon},
            start,velocidad,ds,n_alts
        )

        resumenes=[summarize_route(f) for _,f in results]
        df_comp=pd.DataFrame(resumenes,
            index=[f"Ruta {i+1}" for i in range(len(resumenes))])
        st.subheader("Resumen comparativo")
        st.table(df_comp)

        # Mapa combinado
        m=folium.Map(location=[(ori_lat+des_lat)/2,(ori_lon+des_lon)/2],zoom_start=6)
        colores=["blue","green","orange","purple","red"]
        for i,(c,_) in enumerate(results):
            folium.PolyLine([(lat,lon) for lon,lat in c],
                            color=colores[i],weight=4,
                            tooltip=f"%Alto riesgo: {df_comp.iloc[i]['% Alto riesgo']}").add_to(m)
        m.save("comparison_map.html")
        st.subheader("Mapa combinado de rutas")
        with open("comparison_map.html") as f:
            st.components.v1.html(f.read(),height=600)

##########################################
# EJECUCIÓN
##########################################
authenticator.login(location="main")
status=st.session_state.get("authentication_status")
if status:
    st.write(f"✅ Bienvenido, **{st.session_state['name']}**")
    authenticator.logout("Cerrar sesión","main")
    main_streamlit()
elif status is False:
    st.error("❌ Usuario o contraseña incorrectos")
    st.stop()
else:
    st.info("🔒 Ingresa tus credenciales")
    st.stop()

