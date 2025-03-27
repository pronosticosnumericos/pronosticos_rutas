import os
import glob
import xarray as xr
import pandas as pd
from datetime import date
from numcodecs import Blosc  # Importa Blosc desde numcodecs

def load_dataset(base_dir):
    today = date.today().strftime("%Y-%m-%d")
    files = sorted(glob.glob(os.path.join(base_dir, f"wrfout_d01_{today}*")))
    if not files:
        raise FileNotFoundError(f"No WRF files found in {base_dir}")
    
    # Concatenar a lo largo de la dimensión "Time"
    ds = xr.open_mfdataset(files, combine='nested', concat_dim="Time")
    ds = ds.rename({"Time": "time"})
    
    # Decodificar la variable "Times" si existe, suponiendo el formato "YYYY-MM-DD_HH:MM:SS"
    if "Times" in ds:
        ds["time"] = pd.to_datetime([t.decode() for t in ds["Times"].values],
                                    format="%Y-%m-%d_%H:%M:%S")
    ds = ds.drop_vars("Times", errors="ignore")
    
    # Renombrar dimensiones y asignar coordenadas
    ds = ds.rename_dims({"south_north": "lat", "west_east": "lon"})
    ds = ds.assign_coords({
        "lat": ds["XLAT"].isel(time=0).values[:, 0],
        "lon": ds["XLONG"].isel(time=0).values[0, :]
    })
    return ds

if __name__ == "__main__":
    BASE_DIR = "/home/sig07/WRF/ARWpost/"
    # Lista de variables a conservar (ajusta según tus necesidades)
    VARS = ["T2", "RAINC", "RAINNC", "U10", "V10", "time", "lat", "lon"]

    ds = load_dataset(BASE_DIR)

    # Selecciona las primeras 72 horas (ajusta el slice según lo necesario)
    ds72 = ds.isel(time=slice(0,72))[VARS]

    # Configura la compresión y chunks para cada variable
    encoding = {}
    for var in ds72.data_vars:
        encoding[var] = {
            "compressor": Blosc(cname="zstd", clevel=3, shuffle=2),
            "chunks": (1, 100, 100)  # Ajusta los chunks según la estructura de tu dataset
        }

    output_path = "/home/sig07/pronostico_rutas/wrf_actual.zarr"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    ds72.to_zarr(output_path, mode="w", encoding=encoding)

    print(f"✅ Saved dataset to Zarr: {output_path}")

