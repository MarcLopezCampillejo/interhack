#!/usr/bin/env python3
"""
Optimización de rutas SIN API KEY usando:

1) Geocodificadores públicos sin key:
   - ArcGIS
   - Photon
   - Nominatim como último recurso

2) OSRM público para matriz de tiempos/distancias:
   - Sin API key
   - Válido para demo/hackathon
   - No recomendable para producción

3) OR-Tools si está instalado:
   - Respeta ventanas horarias reales
   - Mete clientes flexibles 00:00 entre clientes con horario cuando caben
   - Usa tiempo de descarga por parada de 20-25 minutos

4) Fallback greedy_gap_fit si OR-Tools no está instalado:
   - Busca el próximo cliente con horario real
   - Intenta meter clientes sin horario 00:00 antes de ese horario
   - Solo los mete si todavía se llega a tiempo al cliente con horario

Instalación mínima:
    pip install pandas requests

Recomendado:
    pip install pandas requests ortools

Uso recomendado:
    py optimizar_rutas_gapfit_app.py --fecha "02/02/2026" --ruta "DA0216"

Listar rutas disponibles:
    py optimizar_rutas_gapfit_app.py --fecha "02/02/2026" --listar-rutas

Con parámetros de almacén y horario:
    py optimizar_rutas_gapfit_app.py --fecha "02/02/2026" --ruta "DA0216" --salida "08:00" --fin "18:00" --depot-lat 41.5400 --depot-lon 2.2139

IMPORTANTE:
- Optimiza por FECHA + Ruta, no por Ruta global.
- El horario 00:00 se trata como cliente FLEXIBLE, no como visita a medianoche.
- Los clientes con horario real actúan como "anclas".
- Los clientes sin horario se insertan en los huecos si caben con descarga de 20-25 min.
"""

import argparse
import json
import math
import re
import sys
import time
from html import escape
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote

import pandas as pd
import requests


# =========================
# CONFIGURACIÓN
# =========================

BASE = Path(__file__).resolve().parent
INPUT_FILE = BASE / "paradas_para_routing.csv"

GEOCACHE_FILE = BASE / "geocache_sin_apikey.csv"

DEPOT_NAME = "ALMACEN"

# Cambia estos valores por la coordenada real del almacén.
# Formato:
#   DEPOT_LAT = latitud
#   DEPOT_LON = longitud
DEPOT_LAT = 41.5400
DEPOT_LON = 2.2139

SHIFT_START = "08:00"
SHIFT_END = "18:00"

# Descarga por parada.
# El usuario pidió 20-25 minutos.
DEFAULT_SERVICE_MIN = 22
SERVICE_MIN_LIMIT = 20
SERVICE_MAX_LIMIT = 25

# OSRM público. No necesita API key.
OSRM_TABLE_URL = "https://router.project-osrm.org/table/v1/driving/{coords}"

# Geocodificadores públicos sin API key.
ARCGIS_GEOCODE_URL = "https://geocode.arcgis.com/arcgis/rest/services/World/GeocodeServer/findAddressCandidates"
PHOTON_GEOCODE_URL = "https://photon.komoot.io/api/"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

# Cambia esto si vas a usar Nominatim más seriamente.
CONTACT_EMAIL = "tu_email@example.com"
USER_AGENT = f"ruta-hackathon-demo/1.0 ({CONTACT_EMAIL})"

GEOCODERS = ["arcgis", "photon", "nominatim"]

# Límite defensivo para OSRM público.
MAX_STOPS_PER_GROUP = 45

GEOCODER_SLEEP_SEC = 1.1
OSRM_SLEEP_SEC = 0.3


# =========================
# UTILIDADES
# =========================

def parse_hhmm_to_min(s: str) -> int:
    m = re.search(r"(\d{1,2}):(\d{2})", str(s))
    if not m:
        raise ValueError(f"Hora inválida: {s}")
    return int(m.group(1)) * 60 + int(m.group(2))


def fmt_min_to_hhmm(m: Optional[float]) -> str:
    if m is None or pd.isna(m):
        return ""
    m = int(round(float(m)))
    return f"{m // 60:02d}:{m % 60:02d}"


def safe_float(x) -> Optional[float]:
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def clean_address(addr: str) -> str:
    if pd.isna(addr):
        return ""
    s = str(addr).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def normalize_fecha_value(value: str) -> str:
    """
    Acepta:
      - 04/02/2026
      - 4/2/2026
      - 2026-02-04

    Devuelve dd/mm/yyyy.
    """
    s = str(value).strip()
    if not s:
        return s

    m = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", s)
    if m:
        y, mo, d = map(int, m.groups())
        return f"{d:02d}/{mo:02d}/{y:04d}"

    m = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{4})", s)
    if m:
        d, mo, y = map(int, m.groups())
        return f"{d:02d}/{mo:02d}/{y:04d}"

    return s


def slug_text(value: str) -> str:
    s = str(value or "").strip().upper()
    s = re.sub(r"[^A-Z0-9]+", "_", s)
    return s.strip("_") or "SIN_VALOR"


def ensure_min_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Asegura columnas necesarias para no romper si el CSV viene incompleto.
    """
    df = df.copy()

    if "FECHA" not in df.columns:
        raise ValueError("El CSV necesita columna FECHA.")
    if "Ruta" not in df.columns:
        raise ValueError("El CSV necesita columna Ruta.")
    if "direccion_completa" not in df.columns:
        raise ValueError("El CSV necesita columna direccion_completa.")

    if "Repartidor" not in df.columns:
        df["Repartidor"] = ""
    if "Vehículo" not in df.columns:
        df["Vehículo"] = ""
    if "Zona" not in df.columns:
        df["Zona"] = ""
    if "Clientes" not in df.columns:
        df["Clientes"] = ""
    if "clientes_en_parada" not in df.columns:
        df["clientes_en_parada"] = df["Clientes"]
    if "cliente_ids" not in df.columns:
        df["cliente_ids"] = ""
    if "orden_original" not in df.columns:
        df["orden_original"] = df.groupby(["FECHA", "Ruta", "Repartidor", "Vehículo"], dropna=False).cumcount() + 1

    if "horario_inicio_min" not in df.columns:
        df["horario_inicio_min"] = pd.NA
    if "horario_fin_min" not in df.columns:
        df["horario_fin_min"] = pd.NA

    # Texto visual de horario
    if "horario_inicio" not in df.columns:
        df["horario_inicio"] = df["horario_inicio_min"].map(fmt_min_to_hhmm)
    if "horario_fin" not in df.columns:
        df["horario_fin"] = df["horario_fin_min"].map(fmt_min_to_hhmm)

    return df


def has_real_time_window(row: pd.Series) -> bool:
    """
    Regla clave pedida:

    - horario_inicio_min = 00:00 / 0 se considera SIN HORARIO real.
    - sin horario = flexible durante la jornada.
    - con horario real = inicio > 0 y fin > inicio.

    Esto evita que 00:00 fuerce el orden al principio.
    """
    start = safe_float(row.get("horario_inicio_min"))
    end = safe_float(row.get("horario_fin_min"))

    if start is None or end is None:
        return False

    start = int(start)
    end = int(end)

    if start <= 0:
        return False

    if end <= start:
        return False

    return True


def effective_window_min(row: pd.Series, shift_start_min: int, shift_end_min: int) -> Tuple[bool, int, int]:
    """
    Devuelve:
      has_horario_real, inicio_efectivo_min, fin_efectivo_min

    Si no tiene horario real, su ventana es toda la jornada.
    """
    real = has_real_time_window(row)
    if not real:
        return False, shift_start_min, shift_end_min

    start = int(safe_float(row.get("horario_inicio_min")))
    end = int(safe_float(row.get("horario_fin_min")))

    # Seguridad: si viene algo absurdo, se relaja.
    if end <= start:
        return False, shift_start_min, shift_end_min

    return True, start, end


def annotate_time_windows(df: pd.DataFrame, shift_start_min: int, shift_end_min: int) -> pd.DataFrame:
    df = df.copy()

    has_list = []
    tw_start = []
    tw_end = []
    tipo = []

    for _, row in df.iterrows():
        has_real, s, e = effective_window_min(row, shift_start_min, shift_end_min)
        has_list.append(has_real)
        tw_start.append(s)
        tw_end.append(e)
        tipo.append("CON_HORARIO_REAL" if has_real else "SIN_HORARIO_00_FLEXIBLE")

    df["tiene_horario_real"] = has_list
    df["ventana_inicio_efectiva_min"] = tw_start
    df["ventana_fin_efectiva_min"] = tw_end
    df["tipo_horario"] = tipo

    return df


def service_minutes(row: pd.Series) -> int:
    """
    Tiempo de descarga por parada.

    Por defecto:
      - mínimo 20 min
      - recomendado/default 22 min
      - máximo 25 min

    Si en el CSV ya existe servicio_min, lo respeta pero lo limita entre 20 y 25.
    """
    explicit = safe_float(row.get("servicio_min"))
    if explicit is not None and explicit > 0:
        return int(round(clamp(explicit, SERVICE_MIN_LIMIT, SERVICE_MAX_LIMIT)))

    total_lineas = safe_float(row.get("total_lineas")) or 0
    total_productos = safe_float(row.get("total_productos")) or 0

    # Pequeño ajuste si hay volumen. No dejamos que pase de 25 porque el usuario pidió 20-25.
    extra = 0
    if total_lineas > 0 or total_productos > 0:
        extra = min(3, math.ceil(total_lineas / 20 + total_productos / 120))

    minutes = DEFAULT_SERVICE_MIN + extra
    return int(round(clamp(minutes, SERVICE_MIN_LIMIT, SERVICE_MAX_LIMIT)))


def minutes_text(x) -> str:
    if x is None or pd.isna(x):
        return ""
    try:
        return f"{float(x):.1f} min"
    except Exception:
        return str(x)


def km_text(x) -> str:
    if x is None or pd.isna(x):
        return ""
    try:
        return f"{float(x):.2f} km"
    except Exception:
        return str(x)


def row_display_name(row: pd.Series) -> str:
    name = row.get("clientes_en_parada")
    if pd.isna(name) or str(name).strip() == "":
        name = row.get("Clientes", row.get("Nombre", "CLIENTE"))
    return str(name).strip()


# =========================
# GEOCODING
# =========================

def load_geocache() -> Dict[str, Tuple[float, float]]:
    """
    Devuelve dict dirección -> (lat, lon)
    """
    if not GEOCACHE_FILE.exists():
        return {}

    cache_df = pd.read_csv(GEOCACHE_FILE)
    cache = {}
    for _, r in cache_df.iterrows():
        addr = clean_address(r.get("direccion_completa", ""))
        lat = safe_float(r.get("lat"))
        lon = safe_float(r.get("lon"))
        if addr and lat is not None and lon is not None:
            cache[addr] = (lat, lon)
    return cache


def save_geocache(cache: Dict[str, Tuple[float, float]]) -> None:
    rows = [
        {"direccion_completa": addr, "lat": lat, "lon": lon}
        for addr, (lat, lon) in sorted(cache.items())
    ]
    pd.DataFrame(rows).to_csv(GEOCACHE_FILE, index=False)


def geocode_address_arcgis(address: str) -> Optional[Tuple[float, float]]:
    address = clean_address(address)
    if not address:
        return None

    params = {
        "SingleLine": address,
        "f": "json",
        "maxLocations": 1,
        "outFields": "Match_addr,Addr_type,Score",
        "countryCode": "ESP",
    }
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
        "Accept-Language": "es",
    }

    try:
        r = requests.get(ARCGIS_GEOCODE_URL, params=params, headers=headers, timeout=30)
        if r.status_code != 200:
            print(f"[ARCGIS ERROR] {r.status_code}: {address}")
            return None

        data = r.json()
        candidates = data.get("candidates", [])
        if not candidates:
            return None

        candidate = candidates[0]
        loc = candidate.get("location", {})
        lon = loc.get("x")
        lat = loc.get("y")
        if lat is None or lon is None:
            return None

        score = candidate.get("score", 0)
        if score is not None and float(score) < 70:
            print(f"[ARCGIS BAJA CONFIANZA {score}] {address}")

        return float(lat), float(lon)

    except Exception as e:
        print(f"[ARCGIS EXCEPTION] {address}: {e}")
        return None


def geocode_address_photon(address: str) -> Optional[Tuple[float, float]]:
    address = clean_address(address)
    if not address:
        return None

    params = {"q": address, "limit": 1, "lang": "es"}
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
        "Accept-Language": "es",
    }

    try:
        r = requests.get(PHOTON_GEOCODE_URL, params=params, headers=headers, timeout=30)
        if r.status_code != 200:
            print(f"[PHOTON ERROR] {r.status_code}: {address}")
            return None

        data = r.json()
        features = data.get("features", [])
        if not features:
            return None

        coords = features[0].get("geometry", {}).get("coordinates", [])
        if len(coords) < 2:
            return None

        lon, lat = coords[0], coords[1]
        return float(lat), float(lon)

    except Exception as e:
        print(f"[PHOTON EXCEPTION] {address}: {e}")
        return None


def geocode_address_nominatim(address: str) -> Optional[Tuple[float, float]]:
    address = clean_address(address)
    if not address:
        return None

    params = {
        "q": address,
        "format": "json",
        "limit": 1,
        "countrycodes": "es",
        "addressdetails": 0,
    }
    headers = {
        "User-Agent": USER_AGENT,
        "From": CONTACT_EMAIL,
        "Accept": "application/json",
        "Accept-Language": "es",
        "Referer": "https://localhost/ruta-hackathon-demo",
    }

    try:
        r = requests.get(NOMINATIM_URL, params=params, headers=headers, timeout=30)
        if r.status_code != 200:
            print(f"[NOMINATIM ERROR] {r.status_code}: {address}")
            return None

        data = r.json()
        if not data:
            return None

        lat = float(data[0]["lat"])
        lon = float(data[0]["lon"])
        return lat, lon

    except Exception as e:
        print(f"[NOMINATIM EXCEPTION] {address}: {e}")
        return None


def geocode_address(address: str) -> Optional[Tuple[float, float]]:
    providers = {
        "arcgis": geocode_address_arcgis,
        "photon": geocode_address_photon,
        "nominatim": geocode_address_nominatim,
    }

    for provider_name in GEOCODERS:
        fn = providers.get(provider_name)
        if fn is None:
            continue

        result = fn(address)
        time.sleep(GEOCODER_SLEEP_SEC)

        if result is not None:
            return result

    print(f"[NO GEO EN NINGUN PROVEEDOR] {address}")
    return None


def add_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Si ya existen lat/lon, las respeta.
    Si no existen, usa geocache y geocodificadores públicos sin API key.
    """
    df = df.copy()

    if "lat" not in df.columns:
        df["lat"] = pd.NA
    if "lon" not in df.columns:
        df["lon"] = pd.NA

    cache = load_geocache()
    geocoded_count = 0

    for idx, row in df.iterrows():
        lat = safe_float(row.get("lat"))
        lon = safe_float(row.get("lon"))
        if lat is not None and lon is not None:
            continue

        addr = clean_address(row.get("direccion_completa", ""))
        if not addr:
            continue

        if addr in cache:
            lat, lon = cache[addr]
        else:
            result = geocode_address(addr)
            if result is None:
                continue
            lat, lon = result
            cache[addr] = (lat, lon)
            geocoded_count += 1

            if geocoded_count % 10 == 0:
                save_geocache(cache)

        df.at[idx, "lat"] = lat
        df.at[idx, "lon"] = lon

    save_geocache(cache)
    return df


# =========================
# OSRM
# =========================

def get_osrm_matrix(points_lon_lat: List[Tuple[float, float]]) -> Tuple[List[List[float]], List[List[float]]]:
    """
    points_lon_lat incluye depósito + paradas.
    Devuelve:
      durations_sec[i][j]
      distances_m[i][j]
    """
    coords = ";".join([f"{lon:.7f},{lat:.7f}" for lon, lat in points_lon_lat])
    url = OSRM_TABLE_URL.format(coords=coords)
    params = {"annotations": "duration,distance"}

    r = requests.get(url, params=params, timeout=60)
    time.sleep(OSRM_SLEEP_SEC)

    if r.status_code != 200:
        raise RuntimeError(f"OSRM error {r.status_code}: {r.text[:500]}")

    data = r.json()
    if data.get("code") != "Ok":
        raise RuntimeError(f"OSRM response not Ok: {data}")

    durations = data.get("durations")
    distances = data.get("distances")

    if durations is None or distances is None:
        raise RuntimeError(f"OSRM no devolvió matriz completa: {data}")

    return durations, distances


# =========================
# OPTIMIZACIÓN
# =========================

def safe_duration_sec(durations_sec: List[List[float]], i: int, j: int) -> int:
    v = durations_sec[i][j]
    if v is None or pd.isna(v):
        return 10**7
    return int(round(v))


def optimize_with_ortools(
    stops_df: pd.DataFrame,
    durations_sec: List[List[float]],
    distances_m: List[List[float]],
    shift_start_min: int,
    shift_end_min: int,
) -> Optional[List[int]]:
    """
    Devuelve lista de índices de paradas en el orden optimizado.
    Los índices son 0..len(stops_df)-1, sin contar depósito.

    OR-Tools hace exactamente lo que necesitas:
    - Los clientes con horario real tienen ventana cerrada.
    - Los clientes 00:00 son flexibles durante la jornada.
    - Como la descarga es 20-25 min, solo mete una parada flexible si cabe.
    """
    try:
        from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    except Exception:
        return None

    n_stops = len(stops_df)
    n_nodes = n_stops + 1

    service = [0] + [int(v) * 60 for v in stops_df["servicio_min"].tolist()]

    manager = pywrapcp.RoutingIndexManager(n_nodes, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return safe_duration_sec(durations_sec, from_node, to_node) + service[from_node]

    transit_cb = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_cb)

    horizon_sec = max(24 * 3600, shift_end_min * 60 + 4 * 3600)
    routing.AddDimension(
        transit_cb,
        4 * 3600,      # permite esperar si llega temprano a una ventana horaria
        horizon_sec,
        False,
        "Time",
    )

    time_dim = routing.GetDimensionOrDie("Time")

    depot_start = routing.Start(0)
    depot_end = routing.End(0)

    time_dim.CumulVar(depot_start).SetRange(shift_start_min * 60, shift_end_min * 60)
    time_dim.CumulVar(depot_end).SetRange(shift_start_min * 60, horizon_sec)

    for stop_i, (_, row) in enumerate(stops_df.iterrows(), start=1):
        start_min = int(row["ventana_inicio_efectiva_min"])
        end_min = int(row["ventana_fin_efectiva_min"])
        index = manager.NodeToIndex(stop_i)
        time_dim.CumulVar(index).SetRange(start_min * 60, end_min * 60)

    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_params.time_limit.FromSeconds(15)

    solution = routing.SolveWithParameters(search_params)
    if solution is None:
        return None

    order = []
    index = routing.Start(0)

    while not routing.IsEnd(index):
        node = manager.IndexToNode(index)
        if node != 0:
            order.append(node - 1)
        index = solution.Value(routing.NextVar(index))

    return order


def simulate_visit(
    stops_df: pd.DataFrame,
    durations_sec: List[List[float]],
    current_node: int,
    current_time_sec: float,
    stop_i: int,
) -> Optional[Dict[str, float]]:
    """
    Simula visitar una parada desde el nodo actual.

    Retorna None si no hay tiempo de viaje válido.
    """
    node = stop_i + 1
    travel = durations_sec[current_node][node]
    if travel is None or pd.isna(travel):
        return None

    arrival = current_time_sec + float(travel)
    row = stops_df.iloc[stop_i]

    start_sec = int(row["ventana_inicio_efectiva_min"]) * 60
    end_sec = int(row["ventana_fin_efectiva_min"]) * 60
    wait = max(0, start_sec - arrival)
    service_start = arrival + wait
    service_end = service_start + int(row["servicio_min"]) * 60
    late = max(0, service_end - end_sec)

    return {
        "node": node,
        "travel_sec": float(travel),
        "arrival_sec": arrival,
        "wait_sec": wait,
        "service_start_sec": service_start,
        "service_end_sec": service_end,
        "late_sec": late,
    }


def can_fit_before_anchor(
    stops_df: pd.DataFrame,
    durations_sec: List[List[float]],
    current_node: int,
    current_time_sec: float,
    flex_i: int,
    anchor_i: int,
) -> Optional[Dict[str, float]]:
    """
    Regla pedida:
    intenta meter una parada SIN HORARIO antes de una parada CON HORARIO.

    Solo devuelve resultado si:
      actual -> flexible
      descarga flexible 20-25 min
      flexible -> cliente con horario
      llegada/servicio del cliente con horario no rompe su ventana

    Si no cabe, devuelve None.
    """
    first = simulate_visit(stops_df, durations_sec, current_node, current_time_sec, flex_i)
    if first is None:
        return None

    flex_node = flex_i + 1
    after_flex_time = first["service_end_sec"]

    second = simulate_visit(stops_df, durations_sec, flex_node, after_flex_time, anchor_i)
    if second is None:
        return None

    if second["late_sec"] > 0:
        return None

    # Detour aproximado:
    # actual -> flex -> anchor comparado con actual -> anchor
    direct = durations_sec[current_node][anchor_i + 1]
    if direct is None or pd.isna(direct):
        direct = 10**7
    detour = first["travel_sec"] + second["travel_sec"] - float(direct)

    return {
        "flex_service_end_sec": first["service_end_sec"],
        "anchor_service_start_sec": second["service_start_sec"],
        "anchor_wait_sec": second["wait_sec"],
        "detour_sec": detour,
        "travel_to_flex_sec": first["travel_sec"],
    }


def optimize_greedy_gap_fit(
    stops_df: pd.DataFrame,
    durations_sec: List[List[float]],
    shift_start_min: int,
    shift_end_min: int,
) -> List[int]:
    """
    Fallback sin OR-Tools.

    Lógica:
    1) Detecta el próximo cliente CON HORARIO REAL.
    2) Antes de ir a ese cliente, prueba si cabe algún cliente SIN HORARIO 00:00.
    3) Lo mete solo si todavía se llega al cliente con horario dentro de su ventana.
    4) Si no cabe ninguno, va al cliente con horario.
    5) Cuando ya no quedan horarios, mete los flexibles por cercanía.

    Esto corrige el fallo típico de ordenar todo por zona/hora y olvidarse de huecos reales.
    """
    remaining = set(range(len(stops_df)))
    order = []

    current_node = 0
    current_time = shift_start_min * 60

    while remaining:
        timed_remaining = sorted(
            [i for i in remaining if bool(stops_df.iloc[i]["tiene_horario_real"])],
            key=lambda i: (
                int(stops_df.iloc[i]["ventana_inicio_efectiva_min"]),
                int(stops_df.iloc[i]["ventana_fin_efectiva_min"]),
                safe_duration_sec(durations_sec, current_node, i + 1),
            ),
        )

        anchor_i = timed_remaining[0] if timed_remaining else None

        # Si hay próximo cliente con horario, intenta rellenar el hueco con uno flexible.
        if anchor_i is not None:
            flexible_remaining = [i for i in remaining if not bool(stops_df.iloc[i]["tiene_horario_real"])]

            candidates = []
            for flex_i in flexible_remaining:
                fit = can_fit_before_anchor(
                    stops_df=stops_df,
                    durations_sec=durations_sec,
                    current_node=current_node,
                    current_time_sec=current_time,
                    flex_i=flex_i,
                    anchor_i=anchor_i,
                )
                if fit is None:
                    continue

                # Score: menor desvío, menor viaje y menor espera futura.
                score = (
                    fit["detour_sec"] * 1.4
                    + fit["travel_to_flex_sec"]
                    + fit["anchor_wait_sec"] * 0.15
                )
                candidates.append((score, flex_i, fit))

            if candidates:
                candidates.sort(key=lambda x: x[0])
                _, chosen_i, visit = candidates[0]
                order.append(chosen_i)
                remaining.remove(chosen_i)
                current_node = chosen_i + 1
                current_time = visit["flex_service_end_sec"]
                continue

            # No cabe flexible antes del horario. Vamos al cliente con horario.
            visit_anchor = simulate_visit(stops_df, durations_sec, current_node, current_time, anchor_i)
            if visit_anchor is None:
                # Si OSRM no tiene ruta, lo mandamos al final por fallback.
                remaining.remove(anchor_i)
                order.append(anchor_i)
                current_node = anchor_i + 1
                continue

            order.append(anchor_i)
            remaining.remove(anchor_i)
            current_node = anchor_i + 1
            current_time = visit_anchor["service_end_sec"]
            continue

        # Si ya no quedan clientes con horario, hacemos nearest neighbor con los flexibles restantes.
        best = None
        for i in remaining:
            visit = simulate_visit(stops_df, durations_sec, current_node, current_time, i)
            if visit is None:
                continue
            score = visit["late_sec"] * 1000 + visit["travel_sec"]
            candidate = (score, i, visit)
            if best is None or candidate < best:
                best = candidate

        if best is None:
            # Último fallback por orden original.
            rest = sorted(
                list(remaining),
                key=lambda i: safe_float(stops_df.iloc[i].get("orden_original")) or 99999,
            )
            order.extend(rest)
            break

        _, chosen_i, visit = best
        order.append(chosen_i)
        remaining.remove(chosen_i)
        current_node = chosen_i + 1
        current_time = visit["service_end_sec"]

    return order


def build_route_output(
    group: pd.DataFrame,
    order: List[int],
    durations_sec: List[List[float]],
    distances_m: List[List[float]],
    shift_start_min: int,
) -> pd.DataFrame:
    rows = []

    current_node = 0
    current_time_sec = shift_start_min * 60
    accumulated_drive_sec = 0
    accumulated_dist_m = 0

    for seq, stop_i in enumerate(order, start=1):
        node = stop_i + 1
        row = group.iloc[stop_i].copy()

        travel_sec = durations_sec[current_node][node]
        dist_m = distances_m[current_node][node]

        if travel_sec is None or pd.isna(travel_sec):
            travel_sec = 0
        if dist_m is None or pd.isna(dist_m):
            dist_m = 0

        arrival_sec = current_time_sec + float(travel_sec)

        tw_start_sec = int(row["ventana_inicio_efectiva_min"]) * 60
        tw_end_sec = int(row["ventana_fin_efectiva_min"]) * 60
        wait_sec = max(0, tw_start_sec - arrival_sec)

        eta_sec = arrival_sec + wait_sec
        service_sec = int(row["servicio_min"]) * 60
        departure_sec = eta_sec + service_sec

        late_sec = max(0, departure_sec - tw_end_sec)

        accumulated_drive_sec += float(travel_sec)
        accumulated_dist_m += float(dist_m)

        row["orden_optimizado"] = seq
        row["eta_hora"] = fmt_min_to_hhmm(eta_sec / 60)
        row["salida_estimada"] = fmt_min_to_hhmm(departure_sec / 60)
        row["tiempo_desde_anterior_min"] = round(float(travel_sec) / 60, 1)
        row["distancia_desde_anterior_km"] = round(float(dist_m) / 1000, 2)
        row["tiempo_conduccion_acumulado_min"] = round(accumulated_drive_sec / 60, 1)
        row["distancia_acumulada_km"] = round(accumulated_dist_m / 1000, 2)
        row["espera_min"] = round(wait_sec / 60, 1)
        row["retraso_min"] = round(late_sec / 60, 1)
        row["servicio_min"] = int(row["servicio_min"])
        row["ventana_inicio_efectiva"] = fmt_min_to_hhmm(row["ventana_inicio_efectiva_min"])
        row["ventana_fin_efectiva"] = fmt_min_to_hhmm(row["ventana_fin_efectiva_min"])

        rows.append(row)

        current_node = node
        current_time_sec = departure_sec

    return pd.DataFrame(rows)


def optimize_one_group(group: pd.DataFrame, group_key) -> pd.DataFrame:
    group = group.copy().reset_index(drop=True)

    shift_start_min = parse_hhmm_to_min(SHIFT_START)
    shift_end_min = parse_hhmm_to_min(SHIFT_END)

    group["servicio_min"] = group.apply(service_minutes, axis=1)
    group = annotate_time_windows(group, shift_start_min, shift_end_min)

    if len(group) > MAX_STOPS_PER_GROUP:
        print(f"[SKIP OPTIMIZATION] Grupo {group_key} tiene {len(group)} paradas. Límite demo: {MAX_STOPS_PER_GROUP}.")
        group = group.sort_values(["Zona", "ventana_inicio_efectiva_min", "orden_original"], na_position="last").copy()
        group["orden_optimizado"] = range(1, len(group) + 1)
        group["estado_optimizacion"] = "demasiadas_paradas_para_osrm_publico"
        return group

    points = [(DEPOT_LON, DEPOT_LAT)]
    for _, r in group.iterrows():
        points.append((float(r["lon"]), float(r["lat"])))

    durations_sec, distances_m = get_osrm_matrix(points)

    order = optimize_with_ortools(
        group,
        durations_sec,
        distances_m,
        shift_start_min,
        shift_end_min,
    )

    if order is None:
        order = optimize_greedy_gap_fit(
            group,
            durations_sec,
            shift_start_min,
            shift_end_min,
        )
        optimizer = "greedy_gap_fit_20_25_min_sin_ortools"
    else:
        optimizer = "ortools_time_windows_gapfit_20_25_min"

    out = build_route_output(
        group,
        order,
        durations_sec,
        distances_m,
        shift_start_min,
    )

    out["estado_optimizacion"] = optimizer
    return out


# =========================
# SALIDAS PARA INTERFAZ
# =========================

def build_output_paths(base: Path, fecha: Optional[str], ruta: Optional[str]) -> Dict[str, Path]:
    fecha_slug = slug_text(str(fecha).replace("/", "")) if fecha else "TODAS_FECHAS"
    ruta_slug = slug_text(ruta) if ruta else "TODAS_RUTAS"
    prefix = f"{fecha_slug}_{ruta_slug}"
    return {
        "csv": base / f"rutas_optimizadas_gapfit_{prefix}.csv",
        "summary": base / f"resumen_rutas_optimizadas_gapfit_{prefix}.csv",
        "txt": base / f"recorrido_gapfit_{prefix}.txt",
        "html": base / f"recorrido_gapfit_{prefix}.html",
        "json": base / f"recorrido_gapfit_{prefix}.json",
        "gmaps": base / f"links_google_maps_gapfit_{prefix}.csv",
    }


def available_routes_for_date(df: pd.DataFrame, fecha: str) -> pd.DataFrame:
    fecha_norm = normalize_fecha_value(fecha)
    dff = df[df["FECHA"].astype(str).map(normalize_fecha_value) == fecha_norm].copy()
    if dff.empty:
        return pd.DataFrame(columns=["FECHA", "Ruta", "paradas", "con_horario", "sin_horario_00", "Repartidor", "Vehículo"])

    shift_start_min = parse_hhmm_to_min(SHIFT_START)
    shift_end_min = parse_hhmm_to_min(SHIFT_END)
    dff = annotate_time_windows(dff, shift_start_min, shift_end_min)

    return (
        dff.groupby(["FECHA", "Ruta"], dropna=False)
        .agg(
            paradas=("direccion_completa", "count"),
            con_horario=("tiene_horario_real", "sum"),
            Repartidor=("Repartidor", "first"),
            Vehículo=("Vehículo", "first"),
        )
        .reset_index()
        .assign(sin_horario_00=lambda x: x["paradas"] - x["con_horario"])
        .sort_values(["Ruta"])
    )


def prompt_if_missing(args, df: pd.DataFrame):
    if args.fecha is None:
        fechas = sorted(df["FECHA"].dropna().astype(str).map(normalize_fecha_value).unique())
        print("\nFechas disponibles, primeras 20:")
        print(", ".join(fechas[:20]))
        args.fecha = input("\nEscribe fecha exactamente así dd/mm/aaaa. Ejemplo 04/02/2026: ").strip()

    args.fecha = normalize_fecha_value(args.fecha)

    if args.ruta is None:
        rutas = available_routes_for_date(df, args.fecha)
        if rutas.empty:
            print(f"No encuentro rutas para la fecha {args.fecha}.")
        else:
            print(f"\nRutas disponibles para {args.fecha}:")
            print(rutas.to_string(index=False))
        args.ruta = input("\nEscribe ruta. Ejemplo DR0051: ").strip().upper()

    args.ruta = str(args.ruta).strip().upper()
    return args


def save_recorrido_txt(final: pd.DataFrame, path: Path, fecha: str, ruta: str) -> None:
    lines = []
    lines.append(f"RECORRIDO OPTIMIZADO GAP-FIT - {fecha} - {ruta}")
    lines.append("=" * 90)
    lines.append("Regla aplicada:")
    lines.append("  - Clientes con horario real = anclas.")
    lines.append("  - Clientes con inicio 00:00 = flexibles.")
    lines.append("  - Se intenta meter flexibles entre horarios solo si caben con descarga de 20-25 min.")
    lines.append("")
    lines.append("Sintaxis usada:")
    lines.append(f"  py {Path(sys.argv[0]).name} --fecha \"{fecha}\" --ruta \"{ruta}\"")
    lines.append("")

    group_cols = ["FECHA", "Ruta", "Repartidor", "Vehículo"]

    for key, group in final.groupby(group_cols, dropna=False):
        fecha_g, ruta_g, rep, veh = key
        group = group.sort_values("orden_optimizado")

        total_km = group["distancia_acumulada_km"].max() if "distancia_acumulada_km" in group.columns else ""
        total_min = group["tiempo_conduccion_acumulado_min"].max() if "tiempo_conduccion_acumulado_min" in group.columns else ""
        estado = group["estado_optimizacion"].iloc[0] if "estado_optimizacion" in group.columns else ""
        con_horario = int(group["tiene_horario_real"].sum()) if "tiene_horario_real" in group.columns else 0
        sin_horario = len(group) - con_horario

        lines.append(f"Grupo: FECHA={fecha_g} | RUTA={ruta_g} | Repartidor={rep} | Vehículo={veh}")
        lines.append(f"Estado: {estado}")
        lines.append(f"Paradas: {len(group)} | Con horario: {con_horario} | Sin horario 00:00: {sin_horario}")
        lines.append(f"Total conducción aprox.: {minutes_text(total_min)} | Distancia aprox.: {km_text(total_km)}")
        lines.append("-" * 90)

        for _, row in group.iterrows():
            orden = int(row.get("orden_optimizado", 0))
            cliente = row_display_name(row)
            direccion = clean_address(row.get("direccion_completa", row.get("Dirección", "")))
            zona = row.get("Zona", "")
            eta = row.get("eta_hora", "")
            salida = row.get("salida_estimada", "")
            tprev = row.get("tiempo_desde_anterior_min", "")
            dprev = row.get("distancia_desde_anterior_km", "")
            espera = row.get("espera_min", "")
            retraso = row.get("retraso_min", "")
            servicio = row.get("servicio_min", "")
            tipo_horario = row.get("tipo_horario", "")
            ventana = f"{row.get('ventana_inicio_efectiva', '')}-{row.get('ventana_fin_efectiva', '')}"

            lines.append(f"{orden:02d}. {eta} -> {salida} | {cliente} | Zona {zona} | {tipo_horario}")
            lines.append(f"    Dirección: {direccion}")
            lines.append(
                f"    Desde anterior: {minutes_text(tprev)} / {km_text(dprev)} | "
                f"Descarga: {servicio} min | Espera: {minutes_text(espera)} | "
                f"Retraso: {minutes_text(retraso)} | Ventana usada: {ventana}"
            )

        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def make_google_maps_links(final: pd.DataFrame, path: Path, max_stops_per_link: int = 8) -> pd.DataFrame:
    rows = []
    group_cols = ["FECHA", "Ruta", "Repartidor", "Vehículo"]

    for key, group in final.groupby(group_cols, dropna=False):
        group = group.sort_values("orden_optimizado").copy()

        coords = [(DEPOT_LAT, DEPOT_LON, "ALMACEN")]
        for _, r in group.iterrows():
            lat = safe_float(r.get("lat"))
            lon = safe_float(r.get("lon"))
            if lat is not None and lon is not None:
                coords.append((lat, lon, row_display_name(r)))

        start_idx = 0
        tramo = 1

        while start_idx < len(coords) - 1:
            segment = coords[start_idx:start_idx + max_stops_per_link + 1]
            if len(segment) < 2:
                break

            origin = f"{segment[0][0]},{segment[0][1]}"
            destination = f"{segment[-1][0]},{segment[-1][1]}"
            waypoints = segment[1:-1]

            url = (
                "https://www.google.com/maps/dir/?api=1"
                f"&origin={quote(origin)}"
                f"&destination={quote(destination)}"
                "&travelmode=driving"
            )

            if waypoints:
                wp = "|".join([f"{lat},{lon}" for lat, lon, _ in waypoints])
                url += f"&waypoints={quote(wp)}"

            rows.append({
                "FECHA": key[0],
                "Ruta": key[1],
                "Repartidor": key[2],
                "Vehículo": key[3],
                "tramo": tramo,
                "desde": segment[0][2],
                "hasta": segment[-1][2],
                "num_paradas_intermedias": len(waypoints),
                "url_google_maps": url,
            })

            start_idx += max_stops_per_link
            tramo += 1

    links = pd.DataFrame(rows)
    links.to_csv(path, index=False)
    return links


def save_recorrido_json(final: pd.DataFrame, path: Path) -> None:
    cols_preferred = [
        "FECHA", "Ruta", "Repartidor", "Vehículo", "orden_optimizado",
        "clientes_en_parada", "cliente_ids", "direccion_completa", "Zona",
        "lat", "lon", "eta_hora", "salida_estimada",
        "tiempo_desde_anterior_min", "distancia_desde_anterior_km",
        "tiempo_conduccion_acumulado_min", "distancia_acumulada_km",
        "espera_min", "retraso_min", "servicio_min",
        "tipo_horario", "tiene_horario_real",
        "ventana_inicio_efectiva", "ventana_fin_efectiva",
        "estado_optimizacion",
    ]

    cols = [c for c in cols_preferred if c in final.columns]
    data = (
        final[cols]
        .sort_values(["FECHA", "Ruta", "Repartidor", "Vehículo", "orden_optimizado"])
        .to_dict(orient="records")
    )

    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def save_recorrido_html(final: pd.DataFrame, path: Path, fecha: str, ruta: str) -> None:
    points = []
    table_rows = []

    final_sorted = final.sort_values(["FECHA", "Ruta", "Repartidor", "Vehículo", "orden_optimizado"]).copy()

    for _, row in final_sorted.iterrows():
        lat = safe_float(row.get("lat"))
        lon = safe_float(row.get("lon"))
        if lat is None or lon is None:
            continue

        orden = int(row.get("orden_optimizado", 0))
        cliente = row_display_name(row)
        direccion = clean_address(row.get("direccion_completa", ""))
        eta = str(row.get("eta_hora", ""))
        salida = str(row.get("salida_estimada", ""))
        zona = str(row.get("Zona", ""))
        tipo = str(row.get("tipo_horario", ""))
        servicio = str(row.get("servicio_min", ""))

        points.append({
            "lat": lat,
            "lon": lon,
            "orden": orden,
            "cliente": cliente,
            "direccion": direccion,
            "eta": eta,
            "salida": salida,
            "zona": zona,
            "tipo": tipo,
            "servicio": servicio,
        })

        table_rows.append(
            "<tr>"
            f"<td>{orden}</td>"
            f"<td>{escape(eta)}</td>"
            f"<td>{escape(salida)}</td>"
            f"<td>{escape(cliente)}</td>"
            f"<td>{escape(zona)}</td>"
            f"<td>{escape(tipo)}</td>"
            f"<td>{escape(servicio)} min</td>"
            f"<td>{escape(direccion)}</td>"
            f"<td>{escape(minutes_text(row.get('tiempo_desde_anterior_min')))}</td>"
            f"<td>{escape(km_text(row.get('distancia_desde_anterior_km')))}</td>"
            f"<td>{escape(minutes_text(row.get('espera_min')))}</td>"
            f"<td>{escape(minutes_text(row.get('retraso_min')))}</td>"
            "</tr>"
        )

    depot = {
        "lat": DEPOT_LAT,
        "lon": DEPOT_LON,
        "orden": 0,
        "cliente": DEPOT_NAME,
        "direccion": "Almacén",
        "eta": SHIFT_START,
        "salida": SHIFT_START,
        "zona": "",
        "tipo": "DEPOT",
        "servicio": "0",
    }

    map_points = [depot] + points
    center_lat = points[0]["lat"] if points else DEPOT_LAT
    center_lon = points[0]["lon"] if points else DEPOT_LON

    html = f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>Recorrido gap-fit {escape(fecha)} {escape(ruta)}</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<style>
body {{ font-family: Arial, sans-serif; margin: 0; background: #f7f7f7; color: #222; }}
header {{ padding: 18px 22px; background: #ffffff; border-bottom: 1px solid #ddd; }}
#map {{ height: 58vh; width: 100%; }}
section {{ padding: 18px 22px; }}
table {{ border-collapse: collapse; width: 100%; background: white; font-size: 13px; }}
th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; }}
th {{ background: #f0f0f0; text-align: left; }}
.badge {{ display: inline-block; padding: 3px 8px; border-radius: 999px; background: #eee; margin-right: 5px; }}
.note {{ color: #555; font-size: 13px; }}
.real {{ background: #fff5cc; }}
.flex {{ background: #eaf7ea; }}
</style>
</head>
<body>
<header>
<h2>Recorrido optimizado gap-fit</h2>
<div>
<span class="badge">Fecha: {escape(fecha)}</span>
<span class="badge">Ruta: {escape(ruta)}</span>
<span class="badge">Descarga: {SERVICE_MIN_LIMIT}-{SERVICE_MAX_LIMIT} min</span>
</div>
<p class="note">
Regla: los clientes con horario real actúan como anclas. Los clientes 00:00 se insertan entre horarios solo si caben con descarga de 20-25 min.
La línea del mapa une los puntos en orden; no representa necesariamente la geometría exacta por carretera.
</p>
</header>

<div id="map"></div>

<section>
<h3>Orden de visita</h3>
<table>
<thead>
<tr>
<th>#</th><th>ETA</th><th>Salida</th><th>Cliente</th><th>Zona</th><th>Tipo horario</th><th>Descarga</th><th>Dirección</th><th>Tiempo anterior</th><th>Km anterior</th><th>Espera</th><th>Retraso</th>
</tr>
</thead>
<tbody>
{''.join(table_rows)}
</tbody>
</table>
</section>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const points = {json.dumps(map_points, ensure_ascii=False)};
const map = L.map('map').setView([{center_lat}, {center_lon}], 11);

L.tileLayer('https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
  maxZoom: 19,
  attribution: '&copy; OpenStreetMap'
}}).addTo(map);

const latlngs = [];

points.forEach(p => {{
  latlngs.push([p.lat, p.lon]);
  const label = p.orden === 0 ? 'A' : String(p.orden);
  const marker = L.marker([p.lat, p.lon]).addTo(map);
  marker.bindPopup(`<b>${{label}} - ${{p.cliente}}</b><br>${{p.eta}} → ${{p.salida}}<br>${{p.tipo}}<br>Descarga: ${{p.servicio}} min<br>${{p.direccion}}`);
}});

if (latlngs.length > 1) {{
  L.polyline(latlngs).addTo(map);
  map.fitBounds(latlngs, {{padding: [30, 30]}});
}}
</script>
</body>
</html>"""

    path.write_text(html, encoding="utf-8")


# =========================
# CLI / MAIN
# =========================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Optimiza una ruta por FECHA + RUTA. Inserta clientes 00:00 entre clientes con horario si caben con descarga de 20-25 min."
    )

    parser.add_argument("--fecha", help='Fecha a optimizar. Ejemplo: "04/02/2026" o "2026-02-04"')
    parser.add_argument("--ruta", help="Ruta a optimizar. Ejemplo: DR0051 o DA0216")
    parser.add_argument("--input", default=str(INPUT_FILE), help="CSV de entrada. Por defecto: paradas_para_routing.csv junto al script")
    parser.add_argument("--depot-lat", type=float, default=DEPOT_LAT, help="Latitud del almacén")
    parser.add_argument("--depot-lon", type=float, default=DEPOT_LON, help="Longitud del almacén")
    parser.add_argument("--salida", default=SHIFT_START, help='Hora de salida. Ejemplo: "08:00"')
    parser.add_argument("--fin", default=SHIFT_END, help='Hora fin jornada. Ejemplo: "18:00"')
    parser.add_argument("--servicio-default", type=int, default=DEFAULT_SERVICE_MIN, help="Descarga normal por parada. Recomendado: 22")
    parser.add_argument("--servicio-min", type=int, default=SERVICE_MIN_LIMIT, help="Mínimo descarga. Recomendado: 20")
    parser.add_argument("--servicio-max", type=int, default=SERVICE_MAX_LIMIT, help="Máximo descarga. Recomendado: 25")
    parser.add_argument("--listar-rutas", action="store_true", help="Muestra rutas disponibles para una fecha y termina")
    parser.add_argument("--no-preguntar", action="store_true", help="Modo app/API: si falta fecha o ruta, falla en vez de preguntar")
    parser.add_argument("--max-stops", type=int, default=MAX_STOPS_PER_GROUP, help="Máximo de paradas por grupo para OSRM público")

    return parser.parse_args()


def main():
    global INPUT_FILE, DEPOT_LAT, DEPOT_LON, SHIFT_START, SHIFT_END
    global DEFAULT_SERVICE_MIN, SERVICE_MIN_LIMIT, SERVICE_MAX_LIMIT, MAX_STOPS_PER_GROUP

    args = parse_args()

    INPUT_FILE = Path(args.input).expanduser().resolve()
    DEPOT_LAT = args.depot_lat
    DEPOT_LON = args.depot_lon
    SHIFT_START = args.salida
    SHIFT_END = args.fin
    DEFAULT_SERVICE_MIN = args.servicio_default
    SERVICE_MIN_LIMIT = args.servicio_min
    SERVICE_MAX_LIMIT = args.servicio_max
    MAX_STOPS_PER_GROUP = args.max_stops

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"No existe el archivo de entrada:\n  {INPUT_FILE}\n\n"
            "Solución: copia 'paradas_para_routing.csv' en la MISMA carpeta que este script, "
            "o usa --input con la ruta real."
        )

    df = pd.read_csv(INPUT_FILE)
    df = ensure_min_columns(df)

    df["FECHA"] = df["FECHA"].astype(str).map(normalize_fecha_value)
    df["Ruta"] = df["Ruta"].astype(str).str.upper().str.strip()
    df["direccion_completa"] = df["direccion_completa"].map(clean_address)

    if args.fecha is not None:
        args.fecha = normalize_fecha_value(args.fecha)
    if args.ruta is not None:
        args.ruta = str(args.ruta).strip().upper()

    if args.listar_rutas:
        if not args.fecha:
            raise ValueError('Para --listar-rutas debes indicar --fecha. Ejemplo: --fecha "04/02/2026"')
        rutas = available_routes_for_date(df, args.fecha)
        if rutas.empty:
            print(f"No hay rutas para {args.fecha}. Revisa la fecha exacta.")
        else:
            print(rutas.to_string(index=False))
        return

    if args.no_preguntar and (not args.fecha or not args.ruta):
        raise ValueError('Falta --fecha o --ruta. Ejemplo: py script.py --fecha "04/02/2026" --ruta DR0051')

    if not args.fecha or not args.ruta:
        args = prompt_if_missing(args, df)

    fecha = normalize_fecha_value(args.fecha)
    ruta = str(args.ruta).strip().upper()

    print("\nParámetros recibidos:")
    print(f"  Fecha: {fecha}")
    print(f"  Ruta: {ruta}")
    print(f"  Almacén lat/lon: {DEPOT_LAT}, {DEPOT_LON}")
    print(f"  Jornada: {SHIFT_START} - {SHIFT_END}")
    print(f"  Descarga: {SERVICE_MIN_LIMIT}-{SERVICE_MAX_LIMIT} min | default={DEFAULT_SERVICE_MIN} min")
    print("\nSintaxis para repetirlo:")
    print(f'  py {Path(sys.argv[0]).name} --fecha "{fecha}" --ruta "{ruta}"')

    work = df[(df["FECHA"] == fecha) & (df["Ruta"] == ruta)].copy()

    if work.empty:
        rutas = available_routes_for_date(df, fecha)
        msg = f"No quedan datos para FECHA={fecha} y Ruta={ruta}."
        if not rutas.empty:
            msg += "\nRutas disponibles para esa fecha:\n" + rutas.to_string(index=False)
        raise ValueError(msg)

    shift_start_min = parse_hhmm_to_min(SHIFT_START)
    shift_end_min = parse_hhmm_to_min(SHIFT_END)
    tmp = annotate_time_windows(work, shift_start_min, shift_end_min)

    print("\nParadas seleccionadas:")
    print(f"  Total: {len(tmp)}")
    print(f"  Con horario real: {int(tmp['tiene_horario_real'].sum())}")
    print(f"  Sin horario 00:00 flexible: {len(tmp) - int(tmp['tiene_horario_real'].sum())}")

    # Ya filtrado por fecha+ruta. Recién ahora geocodificamos.
    work = add_coordinates(work)

    output_paths = build_output_paths(INPUT_FILE.parent, fecha, ruta)
    ungeocoded_file = INPUT_FILE.parent / f"paradas_no_geocodificadas_gapfit_{slug_text(fecha.replace('/', ''))}_{slug_text(ruta)}.csv"

    no_geo = work[work["lat"].isna() | work["lon"].isna()].copy()
    if not no_geo.empty:
        no_geo.to_csv(ungeocoded_file, index=False)
        print(f"[WARNING] {len(no_geo)} paradas sin coordenadas. Guardado: {ungeocoded_file}")

    work = work.dropna(subset=["lat", "lon"]).copy()
    if work.empty:
        raise ValueError("Todas las paradas quedaron sin coordenadas. No se puede optimizar.")

    group_cols = ["FECHA", "Ruta", "Repartidor", "Vehículo"]
    results = []

    for key, group in work.groupby(group_cols, dropna=False):
        print(f"\nOptimizando grupo {key} con {len(group)} paradas...")
        try:
            out = optimize_one_group(group, key)
            results.append(out)
        except Exception as e:
            print(f"[ERROR] Grupo {key}: {e}")
            fallback = group.copy()
            fallback["servicio_min"] = fallback.apply(service_minutes, axis=1)
            fallback = annotate_time_windows(fallback, shift_start_min, shift_end_min)
            fallback = fallback.sort_values(["Zona", "ventana_inicio_efectiva_min", "orden_original"], na_position="last").copy()
            fallback["orden_optimizado"] = range(1, len(fallback) + 1)
            fallback["estado_optimizacion"] = f"error_fallback_zona_horario: {str(e)[:120]}"
            results.append(fallback)

    final = pd.concat(results, ignore_index=True)
    final = final.sort_values(group_cols + ["orden_optimizado"]).copy()
    final.to_csv(output_paths["csv"], index=False)

    agg_kwargs = {
        "paradas": ("direccion_completa", "count"),
        "con_horario_real": ("tiene_horario_real", "sum"),
        "estado": ("estado_optimizacion", "first"),
    }

    if "distancia_acumulada_km" in final.columns:
        agg_kwargs["distancia_total_km"] = ("distancia_acumulada_km", "max")
    if "tiempo_conduccion_acumulado_min" in final.columns:
        agg_kwargs["conduccion_total_min"] = ("tiempo_conduccion_acumulado_min", "max")
    if "eta_hora" in final.columns:
        agg_kwargs["hora_primera"] = ("eta_hora", "first")
        agg_kwargs["hora_ultima"] = ("eta_hora", "last")
    if "retraso_min" in final.columns:
        agg_kwargs["retraso_max_min"] = ("retraso_min", "max")

    summary = final.groupby(group_cols, dropna=False).agg(**agg_kwargs).reset_index()
    summary["sin_horario_00"] = summary["paradas"] - summary["con_horario_real"]
    summary.to_csv(output_paths["summary"], index=False)

    save_recorrido_txt(final, output_paths["txt"], fecha, ruta)
    save_recorrido_json(final, output_paths["json"])
    save_recorrido_html(final, output_paths["html"], fecha, ruta)
    links = make_google_maps_links(final, output_paths["gmaps"])

    print("\nLISTO")
    print(f"CSV optimizado: {output_paths['csv']}")
    print(f"Resumen: {output_paths['summary']}")
    print(f"Recorrido TXT: {output_paths['txt']}")
    print(f"Recorrido JSON para interfaz: {output_paths['json']}")
    print(f"Mapa HTML: {output_paths['html']}")
    print(f"Links Google Maps: {output_paths['gmaps']}")
    print(f"Geocache: {GEOCACHE_FILE}")

    if not links.empty:
        print("\nPrimer link de Google Maps:")
        print(links.iloc[0]["url_google_maps"])

    print("\nRecuerda:")
    print("  - CON_HORARIO_REAL = parada ancla.")
    print("  - SIN_HORARIO_00_FLEXIBLE = se mete en huecos si cabe.")
    print("  - Descarga usada por parada: 20-25 minutos.")


if __name__ == "__main__":
    main()
