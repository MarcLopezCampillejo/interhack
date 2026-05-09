import pandas as pd
import unicodedata, re
from difflib import get_close_matches
from pathlib import Path

BASE = Path('/mnt/data')

def norm(s):
    if pd.isna(s):
        return ''
    s = str(s).upper().strip()
    s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    s = re.sub(r'[^A-Z0-9 ]+', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    # remove common legal/noisy words
    for w in ['BAR ', 'RESTAURANT ', 'CAFETERIA ', 'HOSTAL ', 'CERVECERIA ', 'LA ', 'EL ', 'LOS ', 'LES ', 'L ']:
        pass
    return s

def parse_time_value(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    # pandas may read times as datetime.time or strings like 00:00:00
    m = re.search(r'(\d{1,2}):(\d{2})', s)
    if not m:
        return None
    h, mi = int(m.group(1)), int(m.group(2))
    return h*60 + mi

def fmt_minutes(m):
    if m is None or pd.isna(m):
        return ''
    m = int(m)
    return f'{m//60:02d}:{m%60:02d}'

clientes = pd.read_csv(BASE/'clientes.csv')
rutas = pd.read_csv(BASE/'rutas.csv')

# Load delivery windows if present
try:
    horarios = pd.read_excel(BASE/'Horarios Entrega.XLSX')
    horarios['norm_nombre'] = horarios['Nombre 1'].map(norm)
    horarios['start_min'] = horarios['Horario inicia a'].map(parse_time_value)
    horarios['end_min'] = horarios['Horario termina a'].map(parse_time_value)
    # take earliest opening per client name, ignoring always-open 00:00 when other data is not useful
    hgrp = (horarios.groupby('norm_nombre', as_index=False)
            .agg(horario_inicio_min=('start_min','min'), horario_fin_min=('end_min','max')))
except Exception:
    hgrp = pd.DataFrame(columns=['norm_nombre','horario_inicio_min','horario_fin_min'])

clientes['norm_nombre'] = clientes['Nombre'].map(norm)
rutas['norm_cliente'] = rutas['Clientes'].map(norm)
clientes_dedup = clientes.drop_duplicates('norm_nombre', keep='first')

# Direct merge by normalized client name
merged = rutas.merge(clientes_dedup[['ClienteID','Nombre','Dirección','Zona','Horario','norm_nombre']],
                    left_on='norm_cliente', right_on='norm_nombre', how='left')

# Fill unmatched by fuzzy matching against clientes names
name_to_row = clientes_dedup.set_index('norm_nombre')
known_names = list(name_to_row.index)
for idx, row in merged[merged['ClienteID'].isna()].iterrows():
    nm = row['norm_cliente']
    match = get_close_matches(nm, known_names, n=1, cutoff=0.88)
    if match:
        r = name_to_row.loc[match[0]]
        merged.loc[idx, ['ClienteID','Nombre','Dirección','Zona','Horario','norm_nombre']] = [r['ClienteID'], r['Nombre'], r['Dirección'], r['Zona'], r['Horario'], match[0]]

# Add horarios by normalized name, first from exact client name then route name
merged = merged.merge(hgrp, left_on='norm_cliente', right_on='norm_nombre', how='left', suffixes=('','_hor'))
# If client file has Horario, keep it as text; otherwise use computed window
merged['horario_inicio'] = merged['horario_inicio_min'].map(fmt_minutes)
merged['horario_fin'] = merged['horario_fin_min'].map(fmt_minutes)

# Compute order heuristic. Zone first, then earliest horario, then original order.
# Missing zone last inside route, missing time treated as very flexible.
merged['orden_original'] = merged.groupby(['Ruta','Repartidor','Vehículo']).cumcount() + 1
merged['zona_sort'] = merged['Zona'].fillna('ZZZ')
merged['hora_sort'] = merged['horario_inicio_min'].fillna(9999)
# add small priority: clients with time windows before no-time clients
merged['has_window'] = merged['horario_inicio_min'].notna().astype(int)

ordered = merged.sort_values(['Ruta','Repartidor','Vehículo','zona_sort','hora_sort','orden_original']).copy()
ordered['orden_recomendado'] = ordered.groupby(['Ruta','Repartidor','Vehículo']).cumcount() + 1
ordered['criterio_orden'] = 'zona -> horario_inicio -> orden_actual'

cols = ['Ruta','Repartidor','Vehículo','orden_recomendado','orden_original','Clientes','ClienteID','Nombre','Dirección','Zona','horario_inicio','horario_fin','criterio_orden']
ordered[cols].to_csv(BASE/'rutas_ordenadas_paso1.csv', index=False)

# Create a demo route: choose DR0054 if exists, otherwise largest route
if 'DR0054' in set(ordered['Ruta'].astype(str)):
    demo_route='DR0054'
else:
    demo_route = ordered['Ruta'].value_counts().index[0]

demo = ordered[ordered['Ruta'].astype(str)==str(demo_route)][cols]
demo.to_csv(BASE/f'ruta_demo_{demo_route}_paso1.csv', index=False)

# Summary
summary = (ordered.groupby('Ruta')
           .agg(num_clientes=('Clientes','count'), repartidor=('Repartidor','first'), vehiculo=('Vehículo','first'), zonas=('Zona', lambda x: ', '.join(sorted(set([str(v) for v in x.dropna().unique()]))[:6])))
           .reset_index()
           .sort_values('num_clientes', ascending=False))
summary.to_csv(BASE/'resumen_rutas_paso1.csv', index=False)

print('Generated rutas_ordenadas_paso1.csv', ordered.shape)
print('Demo route:', demo_route, 'clients:', len(demo))
print(demo.head(20).to_string(index=False))
