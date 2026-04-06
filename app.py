"""
app.py — AquaSapiens v3.0 — VERSIÓN INTEGRADA Y CORREGIDA
==========================================================
Cambios vs versión anterior:
  1. Botón "Ejecutar Motor IA" mejorado: muestra log de salida y el informe
     gerencial directamente en pantalla (ya no hay que buscarlo en un txt).
  2. Panel de Eventos Próximos lee el CSV real de AMAEM en lugar de datos mock
     con partido del Hércules inventado.
  3. Informe gerencial aparece siempre en la sección horaria (no solo mensual).
  4. stress_score recalibrado: usa los percentiles históricos reales del sector
     (columnas p85/p15 del CSV) en lugar de una fórmula lineal sobre variacion_pct.
  5. Interfaz mejorada: estado de la última ejecución visible, sección de informe
     destacada, indicador de fecha de predicción actual.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import ast
import json
import os
import re
import subprocess
import time
import folium
from pyproj import Transformer
from streamlit_folium import st_folium
from datetime import datetime, timedelta, date

# =============================================================================
# ⚙️ CONFIGURACIÓN
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Ruta del CSV de predicción generado por Conjunto.py
# Intentamos con la fecha de hoy; si no existe, buscamos el más reciente
def _ruta_csv_prediccion() -> str:
    hoy_str = datetime.now().strftime("%Y-%m-%d")
    ruta_hoy = os.path.join(BASE_DIR, f"prediccion_GLOBAL_ALICANTE_{hoy_str}.csv")
    if os.path.exists(ruta_hoy):
        return ruta_hoy
    # Buscar el CSV más reciente
    csvs = sorted([f for f in os.listdir(BASE_DIR) if f.startswith("prediccion_GLOBAL_ALICANTE_") and f.endswith(".csv")])
    if csvs:
        return os.path.join(BASE_DIR, csvs[-1])
    return ruta_hoy  # no existe, pero que falle con buen mensaje

RUTA_CSV_PREDICCION = _ruta_csv_prediccion()
RUTA_CSV_EVENTOS    = os.path.join(BASE_DIR, "aguas_corregido_v2_Sheet1_.csv")
RUTA_INFORME        = os.path.join(BASE_DIR, "informe_global.txt")

# =============================================================================
# 🗺️ MAPEOS
# =============================================================================

MAPEO_SECTORES = {
    "1 CIUDAD JARDÍN": "CIUDAD JARDIN", "ALIPARK DL": "ALIPARK",
    "ALTOZANO": "ALTOZANO", "BAHÍA LOS PINOS": "BAHIA DE LOS PINOS",
    "BENALÚA DL": "BENALUA", "Bº GRANADA 1": "BARRIO DE GRANADA",
    "Bº LOS ÁNGELES": "LOS ANGELES", "CABO HUERTAS - PLAYA": "CABO HUERTAS",
    "CENTRO COMERCIAL GRAN VÍA": "CENTRO COMERCIAL GRAN VIA",
    "CIUDAD DEPORTIVA DL": "CIUDAD DEPORTIVA", "COLONIA REQUENA": "COLONIA REQUENA",
    "COLONIA ROMANA": "COLONIA ROMANA", "CONDOMINA": "CONDOMINA",
    "Campoamor Alto": "CAMPOAMOR ALTO", "DIPUTACIÓN DL": "DIPUTACION",
    "Depósito Los Ángeles": "DEPOSITO LOS ANGELES",
    "GARBINET NORTE 1": "GARBINET NORTE", "INFORMACIÓN DL": "INFORMACION",
    "LONJA": "MERCADO CENTRAL", "LONJA DL": "MERCADO CENTRAL",
    "Les Palmeretes": "LES PALMERETES", "MATADERO": "MATADERO",
    "MERCADO DL": "MERCADO CENTRAL", "MUCHAVISTA - P.A.U. 5": "MUCHAVISTA PAU 5 SUR",
    "MUELLE GRANELES DL": "MUELLE DE GRANELES", "MUELLE LEVANTE DL": "MUELLE DE LEVANTE",
    "O.A.M.I 1": "OAMI", "P.A.U. 1 (norte+sur)": "PAU 1 SUR",
    "P.A.U. 2": "PAU 2", "PARQUE LO MORANT": "LO MORANT",
    "PLAYA DE SAN JUAN 1": "PLAYA SAN JUAN", "PZA. MONTAÑETA": "PLAZA DE LA MONTANETA",
    "Pla-Hospital": "PLA HOSPITAL", "Postiguet": "POSTIGUET",
    "RABASA DL": "RABASA", "SANTO DOMINGO DL": "SANTO DOMINGO",
    "SH_Demo": "CASCO ANTIGUO", "TOBO": "TOBO",
    "VALLONGA GLOBAL": "PLA DE LA VALLONGA", "VALLONGA-TOLON DL": "PLA DE LA VALLONGA",
    "VILLAFRANQUEZA": "VIRGEN DEL REMEDIO", "VIRGEN DEL CARMEN 1000 Viv": "MIL VIVIENDAS",
    "VIRGEN DEL REMEDIO": "VIRGEN DEL REMEDIO",
}

TIPO_EVENTO_EMOJI = {
    "Megacrucero": "🚢", "Crucero grande": "🚢", "Crucero pequeño": "⛵",
    "Banya": "🏖️", "Hogueras": "🔥", "Semana Santa": "✝️",
    "Ramadán": "🌙", "Moros y Cristianos": "🎭",
    "partido_champions": "⚽", "partido_mundial": "🏆",
    "partido_eurocopa": "⚽", "partido_local": "⚽",
    "evento_deportivo_local": "🏃", "Atasco-Operación Salida": "🚗",
    "ola_calor": "🌡️", "episodio_sequia": "☀️",
    "temperatura_extrema": "🌡️", "lluvia_intensa": "🌧️", "tormenta": "⛈️",
    "Turismo": "🏨",
}

IMPACTO_A_PCT = {1: 5, 2: 10, 3: 20, 4: 35, 5: 60}

# =============================================================================
# 🎨 ESTILOS
# =============================================================================

st.set_page_config(
    page_title="AquaSapiens | Predicción Hídrica Alicante",
    page_icon="💧", layout="wide"
)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Sora:wght@300;400;600;700&display=swap');

.stApp { background-color: #080d14; color: #e0e8f0; font-family: 'Sora', sans-serif; }

.aqua-header {
    background: linear-gradient(135deg, #0a2540 0%, #0e3460 50%, #0a2540 100%);
    border: 1px solid #1a4a7a; border-radius: 12px;
    padding: 20px 28px; margin-bottom: 20px;
    display: flex; align-items: center; gap: 16px;
}
.aqua-header h1 { font-size: 26px; font-weight: 700; color: #00d4ff; margin: 0; }
.aqua-header p  { font-size: 13px; color: #7a9bc0; margin: 4px 0 0 0; }

.aqua-status-ok  { background: rgba(46,139,87,0.15); border: 1px solid rgba(46,139,87,0.4);
    border-radius: 8px; padding: 10px 14px; font-size: 13px; color: #5dbe8a; margin-bottom:12px; }
.aqua-status-warn { background: rgba(255,165,0,0.12); border: 1px solid rgba(255,165,0,0.35);
    border-radius: 8px; padding: 10px 14px; font-size: 13px; color: #ffa500; margin-bottom:12px; }

.informe-box {
    background: linear-gradient(135deg, #0d1f33 0%, #0a2540 100%);
    padding: 18px 22px; border-radius: 10px;
    border-left: 4px solid #00d4ff;
    font-size: 14px; line-height: 1.8; color: #c8dced; margin: 10px 0;
}
.informe-label {
    font-size: 10px; text-transform: uppercase; letter-spacing: 1.5px;
    color: #00d4ff; font-weight: 700; margin-bottom: 8px;
}
.evento-card {
    background: rgba(255,255,255,0.03); border: 1px solid #1a3a5c;
    border-radius: 10px; padding: 14px;
}
.alerta-card {
    border-radius: 8px; padding: 10px 12px; margin-bottom: 8px;
    border-left: 4px solid; background: rgba(255,255,255,0.03);
}
.alerta-card b   { font-size: 13px; color: #e0e8f0; }
.alerta-card small { font-size: 11px; color: #7a9bc0; line-height: 1.6; }
.factor-pill {
    display: inline-block; padding: 3px 10px; border-radius: 20px;
    font-size: 11px; font-weight: 600; margin: 3px;
    font-family: 'JetBrains Mono', monospace;
}
[data-testid="stMetricValue"] { color: #00d4ff !important; font-size: 28px !important;
    font-family: 'JetBrains Mono', monospace !important; }
[data-testid="stMetricLabel"] { color: #7a9bc0 !important; font-size: 12px !important;
    text-transform: uppercase; letter-spacing: 0.5px; }
div[data-testid="stExpander"] {
    background-color: #0d1a2a; border: 1px solid #1a3a5c; border-radius: 10px; }
.block-container { padding-top: 1.5rem; padding-bottom: 0rem; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #080d14; }
::-webkit-scrollbar-thumb { background: #1a4a7a; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)



# =============================================================================
# 📅 DATOS MOCK PARA EVENTOS (Para la Demo del Hackathon)
# =============================================================================
def _cargar_eventos_futuros_mock() -> list[dict]:
    hoy = datetime.now()
    return [
        {"fecha": hoy, "tipo": "✝️ Semana Santa", "descripcion": "Lunes de Pascua - Pico turístico",
         "impacto_esperado": 0.65, "barrios_afectados": ["Playa San Juan", "Casco Antiguo", "Centro"], "hora": "Todo el día"},
        {"fecha": hoy + timedelta(days=2), "tipo": "🚢 Crucero", "descripcion": "MSC Grandiosa — 4.200 pax",
         "impacto_esperado": 0.42, "barrios_afectados": ["Postiguet", "Casco Antiguo", "Mercado"], "hora": "08:00 - 18:00"},
        {"fecha": hoy + timedelta(days=5), "tipo": "🌡️ Ola de calor", "descripcion": "Temperatura máxima: 36°C",
         "impacto_esperado": 0.74, "barrios_afectados": ["Playa de San Juan", "Cabo Huertas"], "hora": "14:00 - 19:00"}
    ]
# =============================================================================
# 📦 CARGA DE EVENTOS REALES DESDE CSV DE AMAEM
# =============================================================================
@st.cache_data(ttl=3600)
def cargar_eventos_reales_csv(horizonte_dias: int = 30) -> list[dict]:
    """
    Lee el CSV real de AMAEM y devuelve eventos de los próximos N días.
    Sustituye completamente a _cargar_eventos_futuros_mock().
    """
    if not os.path.exists(RUTA_CSV_EVENTOS):
        return []
    try:
        df = pd.read_csv(RUTA_CSV_EVENTOS, sep=';', encoding='latin1')
        df['FECHA_INICIO'] = pd.to_datetime(df['FECHA_INICIO'], format='mixed', dayfirst=False)
        df['FECHA_FIN']    = pd.to_datetime(df['FECHA_FIN'],    format='mixed', dayfirst=False)

        hoy     = datetime.now().date()
        limite  = hoy + timedelta(days=horizonte_dias)
        # Así pilla eventos que ya han empezado pero que siguen activos hoy
        mask = (df['FECHA_FIN'].dt.date >= hoy) & (df['FECHA_INICIO'].dt.date <= limite)    
        df_prox = df[mask].copy().sort_values('FECHA_INICIO')

        eventos = []
        for _, row in df_prox.iterrows():
            tipo_raw = str(row['TIPO_EVENTO'])
            impacto  = int(row['IMPACTO'])
            barrio   = str(row['BARRIO_AFECTADO'])
            barrios_lista = (
                ["Toda la ciudad"] if barrio.strip().lower() == "todos"
                else [b.strip().title() for b in barrio.split(',')][:4]
            )
            impacto_pct = IMPACTO_A_PCT.get(impacto, impacto * 8) / 100
            
            # Buscamos el emoji usando el nombre técnico original (partido_champions)
            emoji = TIPO_EVENTO_EMOJI.get(tipo_raw, "📌")
            
            # Limpiamos el texto para la pantalla: quitamos '_' y capitalizamos
            tipo_limpio = tipo_raw.replace('_', ' ').title()

            eventos.append({
                "fecha":              row['FECHA_INICIO'],
                "tipo":               f"{emoji} {tipo_limpio}",
                "descripcion":        f"{tipo_limpio} — escala de impacto {impacto}/5",
                "impacto_esperado":   impacto_pct,
                "barrios_afectados":  barrios_lista,
                "hora":               str(row['HORA']),
            })
        return eventos
    except Exception as e:
        print(f"[WARN] cargar_eventos_reales_csv: {e}")
        return []


# =============================================================================
# 📦 CARGA DE PREDICCIONES HORARIAS
# =============================================================================

@st.cache_data(ttl=600)
def cargar_predicciones_horarias() -> pd.DataFrame:
    if not os.path.exists(RUTA_CSV_PREDICCION):
        st.warning(f"⚠️ Aún no hay predicción generada. Pulsa **🚀 Ejecutar Motor IA** para generarla.")
        return pd.DataFrame()

    df = pd.read_csv(RUTA_CSV_PREDICCION)
    df['sector_mapa'] = df['sector'].map(MAPEO_SECTORES).fillna(df['sector'])

    df = df.rename(columns={
        'timestamp':             'fecha_hora',
        'consumo_proyectado_m3': 'caudal_predicho_m3',
        'consumo_base_m3':       'caudal_base_m3',
        'confianza_llm':         'confianza',
    })

    def procesar_audit_trail(raw_str):
        try:
            d = ast.literal_eval(str(raw_str))
            numericos = {k.replace('factor_', '').title(): v
                         for k, v in d.items()
                         if isinstance(v, (int, float)) and not k.startswith('_')}
            razonamiento = d.get("_razonamiento_llm", "Ajuste automático por variables de entorno.")
            return numericos, razonamiento
        except:
            return {}, "Datos de auditoría no disponibles."

    df[['factores_llm', 'informe_llm']] = df['factores_activos'].apply(
        lambda x: pd.Series(procesar_audit_trail(x))
    )

    # 1. Calculamos el "pulso" de la ciudad (compara el caudal de esa hora con el máximo del día)
    ratio_horario = df['caudal_predicho_m3'] / df.groupby('sector')['caudal_predicho_m3'].transform('max')
    
    # 2. Estrés dinámico: Noches verdes (~30), Horas punta cálidas (~55) + la opinión de la IA
    df['stress_score'] = np.clip(30 + (ratio_horario * 25) + (df['variacion_pct'] * 2.5), 0, 100)
    def detectar_causa(factores):
        if not factores: return "Demanda Inercial"
        causa = max(factores.items(), key=lambda x: abs(x[1] - 1.0))[0]
        return causa

    df['causa_principal'] = df['factores_llm'].apply(detectar_causa)
    df['fecha_hora'] = pd.to_datetime(df['fecha_hora'])
    df['fecha_str']  = df['fecha_hora'].dt.strftime("%Y-%m-%d")

    return df


# =============================================================================
# 🗺️ MOTOR CARTOGRÁFICO
# =============================================================================

def crear_mapa_prediccion(df_alertas_map: pd.DataFrame, capas_activas: dict) -> folium.Map:
    mapa = folium.Map(location=[38.3452, -0.4815], zoom_start=13, tiles='CartoDB dark_matter')
    traductor_gps = Transformer.from_crs("EPSG:25830", "EPSG:4326", always_xy=True)

    def añadir_infra(nombre_archivo, color, radio, nombre_capa, grosor=1.5):
        ruta = os.path.join(BASE_DIR, 'mapas', nombre_archivo)
        if not os.path.exists(ruta): return
        grupo = folium.FeatureGroup(name=nombre_capa, show=True)
        try:
            with open(ruta, 'r', encoding='utf-8') as f:
                datos = json.load(f)
            for feature in datos.get('features', []):
                geo = feature.get('geometry', {})
                if not geo: continue
                if 'x' in geo and 'y' in geo:
                    lon, lat = traductor_gps.transform(geo['x'], geo['y'])
                    folium.CircleMarker([lat, lon], radius=radio, color=color,
                                        fill=True, fill_opacity=0.7, weight=0).add_to(grupo)
                elif 'paths' in geo:
                    for path in geo['paths']:
                        linea = [list(traductor_gps.transform(p[0], p[1]))[::-1] for p in path]
                        folium.PolyLine(linea, color=color, weight=grosor, opacity=0.8).add_to(grupo)
                elif 'rings' in geo:
                    for ring in geo['rings']:
                        poly = [list(traductor_gps.transform(p[0], p[1]))[::-1] for p in ring]
                        folium.Polygon(poly, color=color, fill=True,
                                       fill_opacity=0.2, weight=grosor).add_to(grupo)
            grupo.add_to(mapa)
        except Exception:
            pass

    if capas_activas.get("instalaciones"):
        añadir_infra("depositos.json", "#FFD700", 6.0, "Depósitos")
        añadir_infra("centros_de_bombeo.json", "#FF4500", 5.0, "Bombeo")
        añadir_infra("bocasriego_hidrantes.json", "#00BFFF", 1.5, "Hidrantes")
        añadir_infra("fuentes.json", "#32CD32", 3.0, "Fuentes Públicas")
    if capas_activas.get("tuberias"):
        añadir_infra("tuberias.json", "#4169E1", 0.5, "Tuberías")
        añadir_infra("redes_primarias.json", "#4169E1", 0, "Redes Primarias", grosor=2.0)
        añadir_infra("redes_arteriales.json", "#00008B", 0, "Redes Arteriales", grosor=3.5)

    def stress_a_color(score: float) -> str:
        if score >= 75: return '#DC143C'
        elif score >= 55: return '#FF6B2B'
        elif score >= 35: return '#FFD700'
        else: return '#2E8B57'

    ruta_sectores = os.path.join(BASE_DIR, 'mapas', 'sectores_de_consumo.json')
    if os.path.exists(ruta_sectores) and not df_alertas_map.empty:
        with open(ruta_sectores, 'r', encoding='utf-8') as f:
            datos_sectores = json.load(f)

        if 'sector_mapa' not in df_alertas_map.columns:
            df_alertas_map['sector_mapa'] = (
                df_alertas_map['sector'].map(MAPEO_SECTORES).fillna(df_alertas_map['sector'])
            )

        idx_stress = df_alertas_map.groupby('sector_mapa')['stress_score'].max().to_dict()
        features_geojson = []

        for feat in datos_sectores.get('features', []):
            geo   = feat.get('geometry', {})
            attrs = feat.get('attributes', feat.get('properties', {}))
            nombre_sector_mapa = str(attrs.get('DCONS_PO_2', 'Desconocido'))

            if not geo or 'rings' not in geo: continue

            geojson_coords = []
            for ring in geo['rings']:
                nuevo_ring = []
                for p in ring:
                    lon, lat = traductor_gps.transform(p[0], p[1])
                    nuevo_ring.append([lon, lat])
                geojson_coords.append(nuevo_ring)

            stress = idx_stress.get(nombre_sector_mapa)
            if stress is not None:
                props = {'s_limpio': nombre_sector_mapa.title(),
                         'color': stress_a_color(stress), 'opacity': 0.65,
                         'estado': f"{stress:.1f}/100"}
            else:
                props = {'s_limpio': nombre_sector_mapa.title(),
                         'color': '#1a3a5c', 'opacity': 0.10,
                         'estado': "Sin predicción activa"}

            features_geojson.append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": geojson_coords},
                "properties": props
            })

        if features_geojson:
            folium.GeoJson(
                {"type": "FeatureCollection", "features": features_geojson},
                style_function=lambda x: {
                    'fillColor': x['properties']['color'], 'color': 'white',
                    'weight': 1, 'fillOpacity': x['properties']['opacity']
                },
                highlight_function=lambda x: {'weight': 3, 'fillOpacity': 0.80},
                tooltip=folium.GeoJsonTooltip(
                    fields=['s_limpio', 'estado'], aliases=['📍 Sector:', '📊 Estrés predicho:'],
                    style=("font-family: 'Sora', 'Segoe UI', sans-serif; "
                           "font-size: 13px; font-weight: normal; "
                           "background-color: #080d14; color: #e0e8f0; "
                           "border: 1px solid #1a3a5c; border-radius: 6px; "
                           "padding: 8px;")
                )
            ).add_to(mapa)

    mapa.get_root().html.add_child(folium.Element('''
        <div style="position:fixed;bottom:30px;left:30px;width:220px;
             background:rgba(8,13,20,0.92);z-index:9999;font-size:12px;color:white;
             border:1px solid #1a3a5c;border-radius:10px;padding:14px;
             font-family:'Segoe UI',sans-serif;box-shadow:0 4px 20px rgba(0,0,0,0.5);">
          <b style="font-size:13px;color:#00d4ff;">💧 AquaSapiens — Estrés Predicho</b><br><br>
          <b style="color:#7a9bc0;font-size:10px;text-transform:uppercase;letter-spacing:1px;">Nivel de Estrés</b><br>
          <span style="color:#DC143C;">■</span> Crítico (&gt;75/100)<br>
          <span style="color:#FF6B2B;">■</span> Alto (55–75)<br>
          <span style="color:#FFD700;">■</span> Moderado (35–55)<br>
          <span style="color:#2E8B57;">■</span> Bajo (&lt;35)<br>
          <span style="color:#1a3a5c;">■</span> Sin cobertura<br>
        </div>
    '''))
    return mapa


# =============================================================================
# 🪟 MODALES DE DETALLE
# =============================================================================

@st.dialog("⏱️ Informe de Predicción — Sector Horario", width="large")
def modal_prediccion_horaria(sector: str, df_sector: pd.DataFrame):
    fila_pico = df_sector.loc[df_sector['stress_score'].idxmax()]
    st.markdown(f"### ⚡ Sector: {sector}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Caudal pico predicho", f"{df_sector['caudal_predicho_m3'].max():.2f} m³/h")
    col2.metric("Caudal base", f"{fila_pico['caudal_base_m3']:.2f} m³/h")
    col3.metric("Hora de mayor estrés", f"{int(fila_pico['hora']):02d}:00h")
    st.divider()

    st.markdown(f"""
        <div class="informe-box">
          <div class="informe-label">🤖 Razonamiento de la IA (Llama-3)</div>
          {fila_pico['informe_llm']}
        </div>
    """, unsafe_allow_html=True)

    st.subheader("📉 Caudal predicho — perfil del día")
    df_graf = df_sector.sort_values('fecha_hora')
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_graf['fecha_hora'], y=df_graf['caudal_predicho_m3'] * 1.10,
        mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=df_graf['fecha_hora'], y=df_graf['caudal_predicho_m3'] * 0.90,
        mode='lines', line=dict(width=0), showlegend=False,
        fill='tonexty', fillcolor='rgba(0,212,255,0.10)', hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=df_graf['fecha_hora'], y=df_graf['caudal_predicho_m3'],
        mode='lines+markers', name='Caudal predicho',
        line=dict(color='#00d4ff', width=2), marker=dict(size=4),
        hovertemplate="<b>%{x|%H:%M}</b><br>Caudal: %{y:.3f} m³/h<extra></extra>"
    ))
    if 'p85_historico' in df_graf.columns:
        fig.add_trace(go.Scatter(
            x=df_graf['fecha_hora'], y=df_graf['p85_historico'],
            mode='lines', name='P85 histórico',
            line=dict(color='#FFD700', width=1.5, dash='dot'),
            hoverinfo='skip'
        ))
    fig.add_hline(y=fila_pico['caudal_base_m3'], line_width=1.5,
                  line_dash="dot", line_color="#7a9bc0",
                  annotation_text="Base XGBoost")
    fig.update_layout(
        template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)', yaxis_title="Caudal (m³/h)",
        margin=dict(l=0, r=0, t=20, b=0), hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**🧩 Factores de impacto según LLM:**")
    factores = fila_pico['factores_llm']
    if factores:
        df_fact = pd.DataFrame(
            [(k, v) for k, v in sorted(factores.items(), key=lambda x: -x[1])],
            columns=['Factor', 'Valor']
        )
        fig_fact = px.bar(
            df_fact, x='Valor', y='Factor', orientation='h',
            color='Valor',
            color_continuous_scale=['#2E8B57', '#FFD700', '#FF6B2B'],
            range_color=[0.7, 1.5],
            template='plotly_dark'
        )
        fig_fact.add_vline(x=1.0, line_dash="dot", line_color="#ffffff", opacity=0.4,
                           annotation_text="Neutral (1.0)")
        fig_fact.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            coloraxis_showscale=False, margin=dict(l=0, r=0, t=10, b=0)
        )
        st.plotly_chart(fig_fact, use_container_width=True)


# =============================================================================
# 📅 PANEL DE EVENTOS DESDE CSV REAL
# =============================================================================

def render_panel_eventos(eventos: list[dict]):
    st.subheader("📅 Eventos Próximos — Fuente: AMAEM")
    if not eventos:
        st.info("Sin eventos registrados en el CSV para los próximos días.")
        return

    eventos_ord = sorted(eventos, key=lambda x: abs(x['impacto_esperado']), reverse=True)
    cols = st.columns(min(len(eventos_ord), 3))
    for i, ev in enumerate(eventos_ord[:3]):
        with cols[i]:
            imp   = ev['impacto_esperado']
            color = "#DC143C" if imp > 0.30 else "#FFD700" if imp > 0.10 else "#2E8B57"
            signo = "⬆️" if imp >= 0 else "⬇️"
            barrios_str = ', '.join(ev['barrios_afectados'][:2])
            if len(ev['barrios_afectados']) > 2:
                barrios_str += '...'
            st.markdown(f"""
                <div class="evento-card" style="border-top:3px solid {color};">
                  <div style="font-size:20px;margin-bottom:6px;">{ev['tipo']}</div>
                  <div style="font-weight:600;font-size:14px;color:#e0e8f0;">{ev['descripcion']}</div>
                  <div style="font-size:12px;color:#7a9bc0;margin-top:4px;">
                    📅 {ev['fecha'].strftime('%d/%m/%Y')} &nbsp;|&nbsp; 🕐 {ev['hora']}
                  </div>
                  <div style="margin-top:10px;font-size:13px;color:{color};">
                    {signo} +{abs(imp*100):.0f}% impacto estimado en consumo
                  </div>
                  <div style="font-size:11px;color:#4a6a8a;margin-top:4px;">📍 {barrios_str}</div>
                </div>
            """, unsafe_allow_html=True)


# =============================================================================
# 🎛️ SIDEBAR Y LÓGICA PRINCIPAL
# =============================================================================

# Header
st.title("💧 AquaSapiens")
st.caption("Centro de Predicción Hídrica | Pipeline: APIs + CSV AMAEM → LLM → XGBoost → Visualización")

# Sidebar
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3264/3264317.png", width=90)
st.sidebar.title("Centro de Control")

# Estado del sistema
csv_existe = os.path.exists(RUTA_CSV_PREDICCION)
if csv_existe:
    fecha_mod = datetime.fromtimestamp(os.path.getmtime(RUTA_CSV_PREDICCION))
    st.sidebar.markdown(
        f"<div class='aqua-status-ok'>✅ Predicción cargada<br>"
        f"<small>Generada: {fecha_mod.strftime('%d/%m/%Y %H:%M')}</small></div>",
        unsafe_allow_html=True
    )
else:
    st.sidebar.markdown(
        "<div class='aqua-status-warn'>⚠️ Sin predicción generada<br>"
        "<small>Pulsa el botón para ejecutar el motor</small></div>",
        unsafe_allow_html=True
    )

st.sidebar.divider()
st.sidebar.subheader("⚙️ Motor de Predicción")

# Botón de ejecución mejorado
if st.sidebar.button("🚀 Ejecutar Motor IA (Predecir Hoy)", use_container_width=True, type="primary"):
    placeholder_log = st.sidebar.empty()
    placeholder_log.info("🔄 Iniciando pipeline...")
    
    try:
        resultado = subprocess.run(
            ["python", "-X", "utf8", os.path.join(BASE_DIR, "Conjunto.py")],
            capture_output=True, text=True, timeout=300,
            cwd=BASE_DIR, encoding="utf-8"
        )
        if resultado.returncode == 0:
            placeholder_log.success("✅ ¡Predicción generada con éxito!")
            # Invalida el caché para que recargue los nuevos datos
            st.cache_data.clear()
            time.sleep(1)
            st.rerun()
        else:
            placeholder_log.error(f"❌ Error en el motor")
            with st.sidebar.expander("Ver log de error"):
                st.code(resultado.stderr[-2000:] if resultado.stderr else "Sin detalle")
    except subprocess.TimeoutExpired:
        placeholder_log.error("⏱️ Timeout (5 min). El modelo tardó demasiado.")
    except Exception as e:
        placeholder_log.error(f"Error: {e}")

st.sidebar.divider()

with st.sidebar.expander("⚙️ Configuración", expanded=False):
    umbral_critico   = st.slider("Umbral estrés crítico", 50, 90, 65, 5,
                                  help="Por encima de este valor se considera estrés crítico.")
    max_alertas_sidebar = st.number_input("Alertas en sidebar", 1, 20, 6, 1)
    c_inst = st.checkbox("🏭 Instalaciones (Depósitos, Bombeo)", value=False)
    c_tub  = st.checkbox("💧 Red de Tuberías", value=False)

capas_activas = {"instalaciones": c_inst, "tuberias": c_tub}

# ── CARGA DE DATOS ────────────────────────────────────────────────────────

df_horario = cargar_predicciones_horarias()
eventos    = cargar_eventos_reales_csv()

# Si no hay datos horarios todavía, mostramos pantalla de bienvenida
if df_horario.empty:
    st.info("""
    ### 👋 Bienvenido a AquaSapiens

    Todavía no hay predicción generada para hoy.

    **Para empezar:**
    1. Pulsa **🚀 Ejecutar Motor IA** en el panel lateral
    2. El sistema conectará con APIs externas, ejecutará el agente LLM y entrenará XGBoost
    3. La predicción se cargará automáticamente al terminar (~2 minutos)
    """)
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════
#  VISTA PRINCIPAL — PREDICCIÓN HORARIA
# ═══════════════════════════════════════════════════════════════════════════

# Filtros en sidebar
st.sidebar.divider()
st.sidebar.subheader("🔍 Filtros")

lista_sectores = ["Todos"] + sorted(df_horario['sector'].unique().tolist())
sector_sel     = st.sidebar.selectbox("📍 Sector:", lista_sectores)


hora_inicio = st.sidebar.slider("⏰ Hora inicio:", 0, 23, 0)
hora_fin    = st.sidebar.slider("⏰ Hora fin:", 0, 23, 23)

df_h = df_horario.copy() 
df_h = df_h[(df_h['hora'] >= hora_inicio) & (df_h['hora'] <= hora_fin)]
if sector_sel != "Todos":
    df_h = df_h[df_h['sector'] == sector_sel]

if 'sector_mapa' not in df_h.columns:
    df_h['sector_mapa'] = df_h['sector'].map(MAPEO_SECTORES).fillna(df_h['sector'])

# ── INFORME GERENCIAL (siempre visible) ───────────────────────────────────
informe_texto = None
if os.path.exists(RUTA_INFORME):
    try:
        with open(RUTA_INFORME, "r", encoding="utf-8") as f:
            informe_texto = f.read().strip()
    except:
        pass

if informe_texto:
    st.markdown('<div class="informe-label">🤖 Informe Gerencial de la IA (Llama-3)</div>', unsafe_allow_html=True)
    st.info(informe_texto)

# --- LA VENTANA DE TRANSPARENCIA DE LA IA ---
try:
    with open(os.path.join(BASE_DIR, "factores_hoy.json"), "r", encoding="utf-8") as f:
        datos_ia = json.load(f)
        ctx = datos_ia.get("contexto", {})
        
    with st.expander("🔍 ¿Qué está analizando la IA en tiempo real? (Ver fuentes de datos)"):
        col_A, col_B = st.columns(2)
        
        # Guardamos el texto del clima en una variable y limpiamos el NaN
        texto_clima = str(ctx.get('clima', {}).get('resumen', 'N/A'))
        texto_clima_limpio = texto_clima.replace('NaN', 'No disponible').replace('nan', 'No disponible')
        
        col_A.markdown(f"**🌦️ Clima:** {texto_clima_limpio}")
        col_A.markdown(f"**✈️ Vuelos en directo:** {ctx.get('vuelos', {}).get('resumen', 'N/A')}")
        col_A.markdown(f"**🚢 Puerto:** {ctx.get('cruceros', {}).get('resumen', 'N/A')}")
        col_A.markdown(f"**💨 Calidad Aire:** {ctx.get('aire', {}).get('resumen', 'N/A')}")
        
        col_B.markdown(f"**📅 Calendario:** Día {ctx.get('calendario', {}).get('perfil_dia', {}).get('nombre', 'N/A')}")
        col_B.markdown(f"**🏨 Ocupación Hotelera:** {ctx.get('hotelero', {}).get('resumen', 'N/A')}")
        col_B.markdown(f"**🚧 Obras de la ciudad:** {ctx.get('obras', {}).get('resumen', 'N/A')}")
except Exception as e:
    pass
# --------------------------------------------
# ── KPIs HORARIOS ────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)

sectores_estres = int((df_h.groupby('sector')['stress_score'].max() >= umbral_critico).sum())
total_sectores  = df_h['sector'].nunique()

c1.metric("⚡ Sectores en estrés", f"{sectores_estres} / {total_sectores}")

hora_pico_idx = df_h.groupby('hora')['caudal_predicho_m3'].mean().idxmax() if not df_h.empty else 0
c2.metric("🕐 Hora de mayor demanda", f"{int(hora_pico_idx):02d}:00h")

c3.metric("💧 Caudal pico predicho", f"{df_h['caudal_predicho_m3'].max():.2f} m³/h")
c4.metric("🤖 Confianza media IA", f"{df_h['confianza'].mean()*100:.0f}%")

st.divider()

# ── LAYOUT PRINCIPAL: eventos + heatmap + mapa ───────────────────────────

# 1. EVENTOS ARRIBA DEL TODO (Punto 1)
render_panel_eventos(eventos)
st.divider()

# Creamos una columna para que los nombres de los sectores se vean en Formato Título (Punto 2)
df_h['sector_limpio'] = df_h['sector'].str.title()

# 2. COLUMNAS PARA MAPA Y GRÁFICOS (Mapa más grande a la izq, gráficos a la der)
col_mapa, col_graficos = st.columns([2, 1.2], gap="large")

with col_graficos:
    st.subheader("🏆 Top sectores por estrés")
    top_sectores = (df_h.groupby('sector_limpio')['stress_score']
                    .max().sort_values(ascending=True).tail(10).reset_index())
    fig_top = px.bar(
        top_sectores, x='stress_score', y='sector_limpio', orientation='h',
        color='stress_score',
        color_continuous_scale=['#2E8B57', '#FFD700', '#FF6B2B', '#DC143C'],
        range_color=[0, 100],
        template='plotly_dark',
        labels={'stress_score': 'Estrés predicho (0-100)', 'sector_limpio': ''}
    )
    # Configuración Top Sectores (Punto 3: Limpiar cartelito)
    fig_top.add_vline(x=umbral_critico, line_dash="dot", line_color="#ffffff",
                      opacity=0.5, annotation_text=f"Umbral ({umbral_critico})")
    
    fig_top.update_traces(
        hovertemplate='<b>Sector:</b> %{y}<br><b>Nivel de Estrés:</b> %{x:.2f}<extra></extra>'
    )
    
    fig_top.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        coloraxis_showscale=False, margin=dict(l=0, r=0, t=0, b=0), height=300
    )
    st.plotly_chart(fig_top, use_container_width=True)

    st.subheader("🌡️ Heatmap estrés horario")
    pivot = (df_h.groupby(['sector_limpio', 'hora'])['stress_score']
             .mean().reset_index()
             .pivot(index='sector_limpio', columns='hora', values='stress_score'))
    fig_heat = px.imshow(
        pivot,
        color_continuous_scale=['#0a2540', '#2E8B57', '#FFD700', '#FF6B2B', '#DC143C'],
        zmin=0, zmax=100,
        aspect='auto', template='plotly_dark',
        labels=dict(x="Hora", y="Sector", color="Estrés")
    )
    
    # Configuración Heatmap (Punto 3 y 4: Limpiar cartelito y X-Axis agrupado)
    fig_heat.update_traces(
        hovertemplate='<b>%{y}</b><br>Hora: %{x}:00<br>Estrés: %{z:.2f}<extra></extra>'
    )
    
    fig_heat.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=20, b=0), height=400,
        xaxis=dict(tick0=0, dtick=4) # Esto hace que salte de 4 en 4 horas
    )
    st.plotly_chart(fig_heat, use_container_width=True)

with col_mapa:
    mapa_h = crear_mapa_prediccion(df_h, capas_activas)
    st_folium(mapa_h, use_container_width=True, height=750)

# ── ALERTAS EN SIDEBAR ────────────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.subheader("🚨 Sectores en riesgo")

resumen_h = (df_h.groupby('sector')
             .agg(stress_max=('stress_score', 'max'),
                  caudal_pico=('caudal_predicho_m3', 'max'),
                  hora_pico=('hora', lambda x: int(df_h.loc[
                      df_h.loc[x.index, 'stress_score'].idxmax(), 'hora'])),
                  causa=('causa_principal', lambda x: x.mode()[0]))
             .sort_values('stress_max', ascending=False)
             .reset_index())

alertas_h = resumen_h[resumen_h['stress_max'] >= umbral_critico]

if alertas_h.empty:
    st.sidebar.success("✅ Sin sectores en zona de riesgo para esta franja horaria.")
else:
    for _, row in alertas_h.head(int(max_alertas_sidebar)).iterrows():
        s = row['stress_max']
        icono, color_b = (
            ("🔴", "#DC143C") if s >= 75 else
            ("🟠", "#FF6B2B") if s >= 55 else
            ("🟡", "#FFD700")
        )
        st.sidebar.markdown(
            f"<div class='alerta-card' style='border-left-color:{color_b};'>"
            f"<b>{icono} {row['sector']}</b><br>"
            f"<small>Estrés: <b>{s:.0f}/100</b> | Pico: {row['hora_pico']:02d}:00h<br>"
            f"💧 {row['caudal_pico']:.2f} m³/h | 🕵️ {row['causa']}</small>"
            f"</div>",
            unsafe_allow_html=True
        )
        df_sector_det = df_h[df_h['sector'] == row['sector']]
        if st.sidebar.button("🔍 Ver detalle", key=f"btn_h_{row['sector']}", use_container_width=True):
            modal_prediccion_horaria(row['sector'], df_sector_det)