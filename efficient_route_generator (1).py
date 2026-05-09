#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
efficient_route_generator.py

Generador de rutas nuevas eficientes para el reto Damm Smart Truck.

Idea:
- NO solo reordena la ruta existente.
- Genera un nuevo orden de clientes usando una heurística greedy multiobjetivo.
- Prioriza zona + horarios, y usa pedidos/productos como factor operativo secundario.

Pipeline:
clientes -> zonas -> horarios -> ruta -> pedidos -> productos -> score -> ruta eficiente

Uso:
python efficient_route_generator.py \
  --base-dir . \
  --ruta DR0054 \
  --start-time 08:00 \
  --service-minutes 8 \
  --out-dir outputs

Archivos esperados:
- clientes.csv
- rutas.csv
- pedidos.csv
- productos_clasificados.csv

Outputs:
- efficient_route_<RUTA>.csv
- efficient_route_<RUTA>.json
- route_comparison_<RUTA>.csv
"""

import argparse
import json
import math
import re
import unicodedata
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd


# -------------------------
# Utilidades
# -------------------------

def normalize_text(x):
    """Normaliza texto para hacer joins robustos entre CSVs."""
    if pd.isna(x):
        return ""
    x = str(x).strip().upper()
    x = unicodedata.normalize("NFKD", x)
    x = "".join(c for c in x if not unicodedata.combining(c))
    x = re.sub(r"[^A-Z0-9]+", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def find_col(df, candidates):
    """Busca una columna de forma flexible."""
    norm_map = {normalize_text(c): c for c in df.columns}
    for cand in candidates:
        key = normalize_text(cand)
        if key in norm_map:
            return norm_map[key]
    return None


def parse_time_window(value):
    """
    Convierte horarios tipo:
    - '08:00-10:00'
    - '8:00 a 10:30'
    - 'Mañana'
    en minutos desde 00:00.
    """
    if pd.isna(value) or str(value).strip() == "":
        return None, None

    s = str(value).strip().lower()

    if "mañana" in s or "mati" in s or "matí" in s:
        return 8 * 60, 12 * 60
    if "tarde" in s or "tarda" in s:
        return 12 * 60, 16 * 60
    if "noche" in s:
        return 16 * 60, 20 * 60

    times = re.findall(r"(\d{1,2})[:.](\d{2})", s)
    if len(times) >= 2:
        h1, m1 = map(int, times[0])
        h2, m2 = map(int, times[1])
        return h1 * 60 + m1, h2 * 60 + m2
    if len(times) == 1:
        h, m = map(int, times[0])
        return h * 60 + m, h * 60 + m + 60

    return None, None


def minutes_to_hhmm(minutes):
    minutes = int(round(minutes))
    h = (minutes // 60) % 24
    m = minutes % 60
    return f"{h:02d}:{m:02d}"


def safe_bool(x):
    if pd.isna(x):
        return False
    return str(x).strip().lower() in {"si", "sí", "yes", "true", "1", "alta"}


def detect_lat_lon_cols(df):
    lat_candidates = ["lat", "latitude", "latitud", "y"]
    lon_candidates = ["lon", "lng", "longitude", "longitud", "x"]
    lat_col = find_col(df, lat_candidates)
    lon_col = find_col(df, lon_candidates)
    if lat_col and lon_col:
        return lat_col, lon_col
    return None, None


def distance_score(a, b, has_coords):
    """
    Distancia aproximada.
    Si hay coordenadas, usa distancia euclídea.
    Si no, usa penalización por cambio de zona.
    """
    if has_coords:
        ax, ay = a.get("lon", None), a.get("lat", None)
        bx, by = b.get("lon", None), b.get("lat", None)
        if ax is not None and ay is not None and bx is not None and by is not None:
            try:
                return math.sqrt((float(ax) - float(bx))**2 + (float(ay) - float(by))**2) * 1000
            except Exception:
                pass

    # Sin coordenadas: aproximación por clusters de zona
    return 0 if a.get("zona_norm", "") == b.get("zona_norm", "") else 100


# -------------------------
# Carga de datos
# -------------------------

def load_data(base_dir):
    base_dir = Path(base_dir)

    clientes = pd.read_csv(base_dir / "clientes.csv")
    rutas = pd.read_csv(base_dir / "rutas.csv")
    pedidos = pd.read_csv(base_dir / "pedidos.csv")
    productos = pd.read_csv(base_dir / "productos_clasificados.csv")

    return clientes, rutas, pedidos, productos


def prepare_clients(clientes, rutas, pedidos, productos, ruta=None):
    """
    Crea una tabla de clientes candidatos con:
    - zona
    - horario
    - volumen de pedido
    - complejidad logística de producto
    - orden original si existía
    """

    # Columnas flexibles
    cliente_nombre_col = find_col(clientes, ["Nombre", "Cliente", "Clientes"])
    cliente_id_col = find_col(clientes, ["ClienteID", "Cliente ID", "ID"])
    zona_col = find_col(clientes, ["Zona", "Zonas", "DD"])
    horario_col = find_col(clientes, ["Horario", "Horario Servicio", "Franja", "Ventana"])

    ruta_col = find_col(rutas, ["Ruta"])
    rutas_cliente_col = find_col(rutas, ["Clientes", "Cliente", "Nombre"])
    vehiculo_col = find_col(rutas, ["Vehículo", "Vehiculo", "Tractor"])
    repartidor_col = find_col(rutas, ["Repartidor"])

    pedido_cliente_col = find_col(pedidos, ["Cliente", "Clientes", "Nombre"])
    pedido_producto_col = find_col(pedidos, ["Producto", "ProductoID", "Descripción"])
    cantidad_col = find_col(pedidos, ["Cantidad", "Cdad", "Qty"])

    producto_col = find_col(productos, ["Producto", "ProductoID", "Descripción"])
    categoria_col = find_col(productos, ["categoria_logistica", "Categoria", "Categoría"])
    prioridad_col = find_col(productos, ["prioridad_acceso", "Prioridad"])
    lateral_col = find_col(productos, ["requiere_lateral"])
    fragil_col = find_col(productos, ["fragil", "frágil"])
    pesado_col = find_col(productos, ["pesado"])
    retorno_col = find_col(productos, ["retorno_dinamico", "Retornable"])

    clientes = clientes.copy()
    rutas = rutas.copy()
    pedidos = pedidos.copy()
    productos = productos.copy()

    # Normalizaciones
    clientes["cliente_norm"] = clientes[cliente_nombre_col].apply(normalize_text)
    rutas["cliente_norm"] = rutas[rutas_cliente_col].apply(normalize_text)
    pedidos["cliente_norm"] = pedidos[pedido_cliente_col].apply(normalize_text)
    productos["producto_norm"] = productos[producto_col].apply(normalize_text)
    pedidos["producto_norm"] = pedidos[pedido_producto_col].apply(normalize_text)

    if ruta:
        rutas_sel = rutas[rutas[ruta_col].astype(str).str.upper() == str(ruta).upper()].copy()
    else:
        rutas_sel = rutas.copy()

    rutas_sel["orden_original"] = range(1, len(rutas_sel) + 1)

    # Base de clientes: si hay ruta, clientes de esa ruta; si no, todos
    base = rutas_sel[["cliente_norm", rutas_cliente_col, ruta_col, "orden_original"]].copy()
    base = base.rename(columns={rutas_cliente_col: "cliente_nombre", ruta_col: "ruta"})
    base = base.drop_duplicates("cliente_norm", keep="first")

    # Añadir vehículo/repartidor si existen
    if vehiculo_col:
        veh = rutas_sel[["cliente_norm", vehiculo_col]].drop_duplicates("cliente_norm")
        base = base.merge(veh, on="cliente_norm", how="left").rename(columns={vehiculo_col: "vehiculo"})
    if repartidor_col:
        rep = rutas_sel[["cliente_norm", repartidor_col]].drop_duplicates("cliente_norm")
        base = base.merge(rep, on="cliente_norm", how="left").rename(columns={repartidor_col: "repartidor"})

    # Merge clientes
    client_cols = ["cliente_norm"]
    rename_map = {}

    if cliente_id_col:
        client_cols.append(cliente_id_col)
        rename_map[cliente_id_col] = "cliente_id"
    if zona_col:
        client_cols.append(zona_col)
        rename_map[zona_col] = "zona"
    if horario_col:
        client_cols.append(horario_col)
        rename_map[horario_col] = "horario"

    lat_col, lon_col = detect_lat_lon_cols(clientes)
    if lat_col and lon_col:
        client_cols += [lat_col, lon_col]
        rename_map[lat_col] = "lat"
        rename_map[lon_col] = "lon"

    client_info = clientes[client_cols].drop_duplicates("cliente_norm").rename(columns=rename_map)
    base = base.merge(client_info, on="cliente_norm", how="left")

    if "zona" not in base.columns:
        base["zona"] = "SIN_ZONA"
    if "horario" not in base.columns:
        base["horario"] = ""

    base["zona"] = base["zona"].fillna("SIN_ZONA")
    base["zona_norm"] = base["zona"].apply(normalize_text)

    # Horarios
    parsed = base["horario"].apply(parse_time_window)
    base["window_start"] = parsed.apply(lambda x: x[0])
    base["window_end"] = parsed.apply(lambda x: x[1])
    base["window_start"] = base["window_start"].fillna(8 * 60)
    base["window_end"] = base["window_end"].fillna(18 * 60)

    # Enriquecer pedidos con productos
    prod_cols = ["producto_norm"]
    for c in [categoria_col, prioridad_col, lateral_col, fragil_col, pesado_col, retorno_col]:
        if c and c not in prod_cols:
            prod_cols.append(c)

    pedidos_enriched = pedidos.merge(productos[prod_cols], on="producto_norm", how="left")

    # Scores por línea de pedido
    if cantidad_col:
        pedidos_enriched["cantidad_num"] = pd.to_numeric(pedidos_enriched[cantidad_col], errors="coerce").fillna(1)
    else:
        pedidos_enriched["cantidad_num"] = 1

    pedidos_enriched["is_lateral"] = pedidos_enriched[lateral_col].apply(safe_bool) if lateral_col else False
    pedidos_enriched["is_fragil"] = pedidos_enriched[fragil_col].apply(safe_bool) if fragil_col else False
    pedidos_enriched["is_pesado"] = pedidos_enriched[pesado_col].apply(safe_bool) if pesado_col else False
    pedidos_enriched["is_retorno"] = pedidos_enriched[retorno_col].apply(safe_bool) if retorno_col else False

    # Complejidad operativa: secundaria, no manda la ruta salvo empates
    pedidos_enriched["operational_complexity_line"] = (
        pedidos_enriched["cantidad_num"]
        + pedidos_enriched["is_lateral"].astype(int) * 5
        + pedidos_enriched["is_pesado"].astype(int) * 3
        + pedidos_enriched["is_fragil"].astype(int) * 2
        + pedidos_enriched["is_retorno"].astype(int) * 2
    )

    demand = pedidos_enriched.groupby("cliente_norm").agg(
        total_items=("cantidad_num", "sum"),
        num_lines=("cantidad_num", "count"),
        operational_complexity=("operational_complexity_line", "sum"),
        lateral_items=("is_lateral", "sum"),
        fragile_items=("is_fragil", "sum"),
        heavy_items=("is_pesado", "sum"),
        returnable_items=("is_retorno", "sum")
    ).reset_index()

    base = base.merge(demand, on="cliente_norm", how="left")
    for col in ["total_items", "num_lines", "operational_complexity", "lateral_items", "fragile_items", "heavy_items", "returnable_items"]:
        base[col] = base[col].fillna(0)

    # Coord flag
    has_coords = "lat" in base.columns and "lon" in base.columns

    return base, has_coords


# -------------------------
# Heurística greedy de ruta
# -------------------------

def build_efficient_route(clients_df, start_time="08:00", service_minutes=8):
    """
    Greedy multiobjetivo.

    En cada paso elige el siguiente cliente minimizando:
    - cambio de zona / distancia
    - violación de ventana horaria
    - espera
    - complejidad operativa secundaria
    - preservación suave del orden original como fallback

    Sin coordenadas:
    - la zona actúa como cluster geográfico.
    Con coordenadas:
    - usa distancia aproximada.
    """

    clients = clients_df.copy().reset_index(drop=True)

    # Parse start time
    h, m = map(int, start_time.split(":"))
    current_time = h * 60 + m

    has_coords = "lat" in clients.columns and "lon" in clients.columns

    # Empezamos por el cliente con ventana más temprana y más carga operativa
    remaining = clients.to_dict("records")
    route = []

    current = {
        "zona_norm": "",
        "lat": None,
        "lon": None,
    }

    step = 1

    while remaining:
        best_idx = None
        best_score = None
        best_detail = None

        for i, candidate in enumerate(remaining):
            dist = distance_score(current, candidate, has_coords)

            # Tiempo de viaje estimado:
            # sin coords: 4 min mismo cluster, 14 min cambio zona
            # con coords: proxy simple
            if has_coords:
                travel = min(30, max(4, dist * 3))
            else:
                travel = 4 if current.get("zona_norm", "") == candidate.get("zona_norm", "") else 14
                if current.get("zona_norm", "") == "":
                    travel = 0

            arrival = current_time + travel
            ws = candidate.get("window_start", 8 * 60)
            we = candidate.get("window_end", 18 * 60)

            wait = max(0, ws - arrival)
            late = max(0, arrival - we)

            # Prioridad logística secundaria:
            # clientes con más complejidad se sirven algo antes si no rompe zona/horario.
            complexity = float(candidate.get("operational_complexity", 0))
            complexity_bonus = min(15, complexity / 50)

            original_order_penalty = float(candidate.get("orden_original", 0)) * 0.01

            # Score: horario y zona mandan. Productos solo desempatan/ajustan.
            score = (
                dist * 1.00
                + late * 20.00
                + wait * 0.20
                + travel * 1.50
                + original_order_penalty
                - complexity_bonus
            )

            detail = {
                "distance_or_zone_penalty": round(dist, 2),
                "estimated_travel_min": round(travel, 2),
                "arrival_min": round(arrival, 2),
                "wait_min": round(wait, 2),
                "late_min": round(late, 2),
                "complexity_bonus": round(complexity_bonus, 2),
                "score": round(score, 2)
            }

            if best_score is None or score < best_score:
                best_score = score
                best_idx = i
                best_detail = detail

        chosen = remaining.pop(best_idx)

        travel = best_detail["estimated_travel_min"]
        arrival = current_time + travel
        service_start = max(arrival, chosen.get("window_start", 8 * 60))
        departure = service_start + service_minutes

        chosen["new_order"] = step
        chosen["estimated_arrival"] = minutes_to_hhmm(service_start)
        chosen["estimated_departure"] = minutes_to_hhmm(departure)
        chosen["route_score_step"] = best_detail["score"]
        chosen["route_decision_detail"] = json.dumps(best_detail, ensure_ascii=False)

        route.append(chosen)

        current = chosen
        current_time = departure
        step += 1

    route_df = pd.DataFrame(route)

    # Columnas finales legibles
    preferred_cols = [
        "new_order", "ruta", "cliente_nombre", "cliente_id", "zona", "horario",
        "estimated_arrival", "estimated_departure",
        "total_items", "num_lines", "operational_complexity",
        "lateral_items", "fragile_items", "heavy_items", "returnable_items",
        "orden_original", "vehiculo", "repartidor",
        "route_score_step", "route_decision_detail"
    ]
    cols = [c for c in preferred_cols if c in route_df.columns]
    extra = [c for c in route_df.columns if c not in cols and c not in {"cliente_norm", "zona_norm", "window_start", "window_end"}]
    return route_df[cols + extra]


def compare_with_original(route_df):
    comp_cols = [c for c in ["cliente_nombre", "orden_original", "new_order", "zona", "horario", "estimated_arrival"] if c in route_df.columns]
    comp = route_df[comp_cols].copy()
    if "orden_original" in comp.columns:
        comp["order_delta"] = comp["orden_original"] - comp["new_order"]
    return comp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default=".", help="Carpeta con clientes.csv, rutas.csv, pedidos.csv y productos_clasificados.csv")
    parser.add_argument("--ruta", default="DR0054", help="Ruta a optimizar, por ejemplo DR0054. Usa ALL para todas.")
    parser.add_argument("--start-time", default="08:00", help="Hora inicio reparto HH:MM")
    parser.add_argument("--service-minutes", type=int, default=8, help="Minutos estimados de servicio por cliente")
    parser.add_argument("--out-dir", default="outputs", help="Carpeta de salida")
    args = parser.parse_args()

    ruta_arg = None if str(args.ruta).upper() == "ALL" else args.ruta

    clientes, rutas, pedidos, productos = load_data(args.base_dir)
    client_base, has_coords = prepare_clients(clientes, rutas, pedidos, productos, ruta=ruta_arg)

    route_df = build_efficient_route(client_base, start_time=args.start_time, service_minutes=args.service_minutes)
    comparison_df = compare_with_original(route_df)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ruta_name = args.ruta.upper()
    route_csv = out_dir / f"efficient_route_{ruta_name}.csv"
    route_json = out_dir / f"efficient_route_{ruta_name}.json"
    comp_csv = out_dir / f"route_comparison_{ruta_name}.csv"

    route_df.to_csv(route_csv, index=False)
    comparison_df.to_csv(comp_csv, index=False)

    payload = {
        "algorithm": "Greedy multiobjetivo zona/horario con complejidad operativa secundaria",
        "route": ruta_name,
        "has_coordinates": bool(has_coords),
        "criteria_priority": [
            "1. Respetar ventanas horarias",
            "2. Minimizar cambios de zona o distancia",
            "3. Mantener clusters operativos",
            "4. Priorizar carga compleja solo como criterio secundario",
            "5. Usar orden original solo como fallback"
        ],
        "clients": route_df.to_dict(orient="records")
    }
    route_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Ruta eficiente generada: {route_csv}")
    print(f"Comparación con orden original: {comp_csv}")
    print(f"JSON para demo/frontend: {route_json}")


if __name__ == "__main__":
    main()
