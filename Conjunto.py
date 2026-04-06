"""
motor_predictivo_hibrido.py  — VERSIÓN CORREGIDA
=================================================
Correcciones aplicadas sobre la versión original:

  ORIGINAL (bugs):
  1. Producto acumulado de factores → explosión combinatoria (1.10^8 = 2.14x)
  2. Umbral de alerta sobre variacion_pct del multiplicador puro (no del consumo real)
  3. Clamping individual hasta 2.50 demasiado permisivo
  4. factor_calor_acumulado elevado a exponente 1.5 antes de entrar al producto

  CORRECCIONES:
  1. Combinación por MEDIA PONDERADA con pesos por relevancia (no producto)
  2. Cap duro del multiplicador final en ±MAX_VARIACION antes de la alerta
  3. Clamping individual más estricto: [0.70, 1.50]
  4. Calor en playa: peso x2 en la media ponderada, no exponente
  5. Umbrales de alerta relativos al consumo histórico del sector (percentiles)
  6. Fallback explícito a 1.0 si el LLM devuelve factores fuera de rango
"""

import json
import warnings
from datetime import timedelta
import agente_llm

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# ==============================================================================
# CONSTANTES GLOBALES
# ==============================================================================

# BUG FIX #3: clamping individual más estricto
FACTOR_MIN = 0.70   # antes 0.30 — un factor nunca debería bajar el consumo un 70%
FACTOR_MAX = 1.50   # antes 2.50 — un factor aislado no puede doblar el consumo

# BUG FIX #2: cap duro del multiplicador combinado ANTES de calcular la alerta
MULT_CAP_MIN = 0.60  # máximo -40% de variación combinada
MULT_CAP_MAX = 1.40  # máximo +40% de variación combinada

# Umbrales de alerta (sobre la variación porcentual del multiplicador ya capado)
UMBRAL_ESTRES = 15.0   # antes 30% — ahora más sensible y realista
UMBRAL_CAIDA  = -15.0  # antes -30%


# ==============================================================================
# 1. PREPARACIÓN DE DATOS Y FEATURE ENGINEERING
# ==============================================================================

def preparar_datos_ml(ruta_csv: str) -> tuple[pd.DataFrame, LabelEncoder]:
    """
    Carga el CSV, construye features temporales y lags reales por sector.
    Devuelve el DataFrame limpio y el LabelEncoder ajustado.
    """
    print("⚙️  Preparando datos para XGBoost...")
    df = pd.read_csv(ruta_csv)
    df["FECHA_HORA"] = pd.to_datetime(df["FECHA_HORA"])
    df = df.sort_values(["SECTOR", "FECHA_HORA"]).reset_index(drop=True)

    df["Hora"]           = df["FECHA_HORA"].dt.hour
    df["DiaSemana"]      = df["FECHA_HORA"].dt.dayofweek
    df["Mes"]            = df["FECHA_HORA"].dt.month
    df["Es_FinDeSemana"] = df["DiaSemana"].isin([5, 6]).astype(int)

    df["Lag_24h"]  = df.groupby("SECTOR")["CAUDAL_M3"].shift(24)
    df["Lag_168h"] = df.groupby("SECTOR")["CAUDAL_M3"].shift(168)

    le = LabelEncoder()
    df["SECTOR_ENC"] = le.fit_transform(df["SECTOR"])

    # Calculamos percentiles históricos por sector y hora para umbrales de alerta
    df["p85_sector_hora"] = df.groupby(["SECTOR", "Hora"])["CAUDAL_M3"].transform(
        lambda x: x.quantile(0.85)
    )
    df["p15_sector_hora"] = df.groupby(["SECTOR", "Hora"])["CAUDAL_M3"].transform(
        lambda x: x.quantile(0.15)
    )

    cols_eliminar = [c for c in ["METODO"] if c in df.columns]
    df = df.drop(columns=cols_eliminar).dropna()

    print(f"   → {len(df):,} registros listos | {df['SECTOR'].nunique()} sectores | "
          f"rango: {df['FECHA_HORA'].min().date()} → {df['FECHA_HORA'].max().date()}")
    return df, le


# ==============================================================================
# 2. ENTRENAMIENTO CON VALIDACIÓN TEMPORAL (TimeSeriesSplit)
# ==============================================================================

FEATURES = ["Hora", "DiaSemana", "Mes", "Es_FinDeSemana",
            "Lag_24h", "Lag_168h", "SECTOR_ENC"]


def entrenar_modelo(df: pd.DataFrame) -> XGBRegressor:
    """
    Entrena XGBRegressor con TimeSeriesSplit. Devuelve modelo entrenado con todos los datos.
    """
    print("\n🧠 Entrenando XGBoost con validación temporal (TimeSeriesSplit)...")

    X = df[FEATURES]
    y = df["CAUDAL_M3"]

    tscv = TimeSeriesSplit(n_splits=5)
    maes = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        m = XGBRegressor(n_estimators=150, learning_rate=0.1,
                         max_depth=5, random_state=42, n_jobs=-1)
        m.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = m.predict(X.iloc[test_idx])
        mae  = mean_absolute_error(y.iloc[test_idx], pred)
        maes.append(mae)
        print(f"   Fold {fold}: MAE = {mae:.3f} M³")

    print(f"\n✅ MAE medio CV temporal: {np.mean(maes):.3f} ± {np.std(maes):.3f} M³")

    modelo_final = XGBRegressor(n_estimators=150, learning_rate=0.1,
                                max_depth=5, random_state=42, n_jobs=-1)
    modelo_final.fit(X, y)
    print("✅ Modelo final entrenado sobre el dataset completo.")
    return modelo_final


# ==============================================================================
# 3. OBTENCIÓN DE LAGS REALES PARA INFERENCIA
# ==============================================================================

def _obtener_lag_real(df_sector: pd.DataFrame, timestamp: pd.Timestamp,
                      horas_atras: int) -> float:
    ts_objetivo = timestamp - timedelta(hours=horas_atras)
    fila = df_sector[df_sector["FECHA_HORA"] == ts_objetivo]

    if not fila.empty:
        return float(fila["CAUDAL_M3"].iloc[0])

    hora = ts_objetivo.hour
    media = df_sector[df_sector["Hora"] == hora]["CAUDAL_M3"].mean()
    warnings.warn(
        f"⚠️  Lag real no encontrado para {ts_objetivo}. "
        f"Usando media histórica a las {hora:02d}h como fallback."
    )
    return float(media)


# ==============================================================================
# 4. LÓGICA DE FACTORES LLM — CORRECCIÓN CENTRAL
# ==============================================================================

_FACTORES_POR_ZONA = {
    "ZONA_CENTRO": {
        "factor_global", "factor_zona_centro", "factor_cruceros",
        "factor_eventos", "factor_ocupacion_hotelera", "factor_obras_construccion",
        "factor_movilidad_ciudad", "factor_fin_de_semana",
    },
    "ZONA_NORTE": {
        "factor_global", "factor_zona_norte", "factor_eventos",
        "factor_obras_construccion", "factor_movilidad_ciudad", "factor_fin_de_semana",
    },
    "PLAYA_SAN_JUAN": {
        "factor_global", "factor_playa_san_juan", "factor_calor_acumulado",
        "factor_vacaciones_escolares", "factor_vuelos_turismo",
        "factor_ocupacion_hotelera", "factor_fin_de_semana",
    },
}

_FACTORES_POR_TRAMO = {
    "manana": {"factor_franja_manana"},
    "tarde":  {"factor_franja_tarde", "factor_calor_acumulado"},
    "noche":  {"factor_franja_noche"},
}

# BUG FIX #4: pesos por factor en lugar de exponentes ad-hoc
# Factores más relevantes tienen mayor peso en la media ponderada
_PESOS_FACTOR = {
    "factor_global":              2.0,  # siempre tiene alto peso
    "factor_calor_acumulado":     2.5,  # antes se elevaba a ^1.5, ahora doble peso en media
    "factor_eventos":             1.5,
    "factor_ocupacion_hotelera":  1.5,
    "factor_fin_de_semana":       1.0,
    "factor_franja_manana":       0.8,
    "factor_franja_tarde":        1.2,
    "factor_franja_noche":        0.6,
    # resto de factores: peso 1.0 por defecto
}


def hora_a_tramo(hora: int) -> str:
    if 6 <= hora < 14:  return "manana"
    if 14 <= hora < 22: return "tarde"
    return "noche"


def asignar_macrozona(sector: str) -> str:
    s = sector.upper()
    if any(x in s for x in ["PLAYA", "CABO", "CONDOMINA", "MUCHAVISTA"]):
        return "PLAYA_SAN_JUAN"
    elif any(x in s for x in ["CENTRO", "MERCADO", "RAMBLA", "BENALÚA", "ALIPARK", "DIPUTACIÓN"]):
        return "ZONA_CENTRO"
    else:
        return "ZONA_NORTE"


def _sanitizar_factor(valor_raw, nombre: str) -> float:
    """
    Convierte el valor devuelto por el LLM a float con clamping estricto.
    Devuelve 1.0 (neutro) si el valor es inválido.
    """
    try:
        v = float(valor_raw)
        if not np.isfinite(v):
            return 1.0
        return max(FACTOR_MIN, min(FACTOR_MAX, v))
    except (TypeError, ValueError):
        warnings.warn(f"⚠️ Factor '{nombre}' inválido ({valor_raw!r}), usando 1.0")
        return 1.0


def aplicar_factores_llm(
    factores_ia: dict, zona: str, tramo: str, hora: int, es_ramadan: bool = False
) -> tuple[float, dict]:
    """
    BUG FIX #1 — Combina factores LLM mediante MEDIA PONDERADA en lugar de producto.

    Razón: con producto, N factores de 1.10 dan 1.10^N → explosión exponencial.
    Con media ponderada, el resultado es siempre proporcional a la desviación media
    de los factores, independientemente de cuántos haya activos.

    BUG FIX #2 — Cap duro del multiplicador final antes de calcular la alerta.
    BUG FIX #3 — Clamping individual [FACTOR_MIN, FACTOR_MAX] más estricto.
    BUG FIX #4 — factor_calor_acumulado tiene doble peso en media, no exponente ^1.5.
    """
    if zona not in _FACTORES_POR_ZONA:
        raise ValueError(f"Zona '{zona}' desconocida.")
    if tramo not in _FACTORES_POR_TRAMO:
        raise ValueError(f"Tramo '{tramo}' desconocido.")

    activos = _FACTORES_POR_ZONA[zona] | _FACTORES_POR_TRAMO[tramo]
    if es_ramadan and zona == "ZONA_NORTE" and tramo == "noche":
        activos.add("factor_ramadan_nocturno")

    audit_trail = {}
    suma_ponderada = 0.0
    suma_pesos     = 0.0

    for nombre_factor in sorted(activos):
        valor = _sanitizar_factor(factores_ia.get(nombre_factor, 1.0), nombre_factor)
        peso  = _PESOS_FACTOR.get(nombre_factor, 1.0)

        # calor en playa tarde/noche: doble peso en la media (antes era ^1.5 en el producto)
        if nombre_factor == "factor_calor_acumulado" and zona == "PLAYA_SAN_JUAN" and tramo in ("tarde", "noche"):
            peso *= 2.0
            audit_trail["_nota_calor"] = f"Peso x2 en PLAYA {tramo} (era ^1.5 en versión con producto)"

        suma_ponderada += valor * peso
        suma_pesos     += peso
        audit_trail[nombre_factor] = round(valor, 4)

    # BUG FIX #1: multiplicador como media ponderada
    multiplicador_raw = suma_ponderada / suma_pesos if suma_pesos > 0 else 1.0

    # Ajuste por confianza del LLM (igual que antes)
    confianza = max(0.0, min(1.0, float(factores_ia.get("confianza", 1.0))))
    multiplicador_ajustado = 1.0 + (multiplicador_raw - 1.0) * confianza

    # BUG FIX #2: cap duro del multiplicador final
    multiplicador_final = max(MULT_CAP_MIN, min(MULT_CAP_MAX, multiplicador_ajustado))

    audit_trail["_multiplicador_raw"]       = round(multiplicador_raw, 4)
    audit_trail["_confianza_llm"]           = round(confianza, 2)
    audit_trail["_multiplicador_ajustado"]  = round(multiplicador_ajustado, 4)
    audit_trail["_multiplicador_final"]     = round(multiplicador_final, 4)
    audit_trail["_zona"]                    = zona
    audit_trail["_tramo"]                   = tramo
    audit_trail["_hora"]                    = hora
    audit_trail["_razonamiento_llm"]        = factores_ia.get("razonamiento", "")

    return multiplicador_final, audit_trail


# ==============================================================================
# 5. PREDICCIÓN DEL PERFIL COMPLETO DE 24H
# ==============================================================================

def _calcular_alerta(consumo_proyectado: float, consumo_base: float,
                     variacion_pct: float,
                     p85: float, p15: float) -> str:
    """
    BUG FIX #2 mejorado: la alerta combina la variación del multiplicador
    con los percentiles históricos reales del sector.

    - Si la proyección supera el p85 histórico Y hay variación positiva → ESTRÉS
    - Si la proyección cae por debajo del p15 Y hay variación negativa → CAÍDA
    - Si solo el multiplicador supera el umbral pero el consumo es normal → NORMAL
    """
    sobre_percentil_alto = consumo_proyectado > p85
    bajo_percentil_bajo  = consumo_proyectado < p15

    if variacion_pct > UMBRAL_ESTRES and sobre_percentil_alto:
        return "🔴 ESTRÉS"
    elif variacion_pct < UMBRAL_CAIDA and bajo_percentil_bajo:
        return "🔵 CAÍDA"
    elif variacion_pct > UMBRAL_ESTRES:
        # Multiplicador alto pero consumo absoluto dentro del rango histórico → advertencia suave
        return "🟡 VIGILAR"
    else:
        return "🟢 NORMAL"


def predecir_perfil_24h(
    modelo, df_historico: pd.DataFrame, le, sector_objetivo: str,
    fecha_str: str, factores_ia: dict, es_ramadan: bool = False
) -> pd.DataFrame:

    fecha_base = pd.to_datetime(fecha_str).normalize()
    df_sector  = df_historico[df_historico["SECTOR"] == sector_objetivo].copy()

    if df_sector.empty:
        raise ValueError(f"Sector '{sector_objetivo}' no encontrado en histórico.")

    sector_enc = int(le.transform([sector_objetivo])[0])
    zona       = asignar_macrozona(sector_objetivo)
    registros  = []

    for hora in range(24):
        ts    = fecha_base + timedelta(hours=hora)
        tramo = hora_a_tramo(hora)

        lag_24h  = _obtener_lag_real(df_sector, ts, 24)
        lag_168h = _obtener_lag_real(df_sector, ts, 168)

        # Percentiles históricos de esta hora en este sector
        filas_hora = df_sector[df_sector["Hora"] == hora]
        p85 = filas_hora["CAUDAL_M3"].quantile(0.85) if not filas_hora.empty else np.inf
        p15 = filas_hora["CAUDAL_M3"].quantile(0.15) if not filas_hora.empty else 0.0

        fila_futura = pd.DataFrame([{
            "Hora": hora, "DiaSemana": ts.dayofweek, "Mes": ts.month,
            "Es_FinDeSemana": 1 if ts.dayofweek in [5, 6] else 0,
            "Lag_24h": lag_24h, "Lag_168h": lag_168h, "SECTOR_ENC": sector_enc,
        }])

        consumo_base = max(0.1, float(modelo.predict(fila_futura)[0]))

        multiplicador, audit = aplicar_factores_llm(
            factores_ia=factores_ia, zona=zona, tramo=tramo,
            hora=hora, es_ramadan=es_ramadan
        )

        consumo_proyectado = consumo_base * multiplicador
        variacion_pct      = (multiplicador - 1.0) * 100  # variación del ajuste LLM

        alerta = _calcular_alerta(consumo_proyectado, consumo_base, variacion_pct, p85, p15)

        registros.append({
            "hora":                  hora,
            "timestamp":             ts,
            "zona":                  zona,
            "tramo":                 tramo,
            "consumo_base_m3":       round(consumo_base, 3),
            "consumo_proyectado_m3": round(consumo_proyectado, 3),
            "multiplicador_llm":     round(multiplicador, 4),
            "confianza_llm":         audit["_confianza_llm"],
            "variacion_pct":         round(variacion_pct, 2),
            "alerta":                alerta,
            "p85_historico":         round(p85, 3),
            "p15_historico":         round(p15, 3),
            "factores_activos":      {k: v for k, v in audit.items() if not k.startswith("_")},
        })

    return pd.DataFrame(registros)


# ==============================================================================
# 6. REPORTE FINAL
# ==============================================================================

def generar_reporte_gerencial(df_pred, sector, contexto_texto):
    """Envía resultados a Groq para redactar el parte operativo en lenguaje natural."""
    from groq import Groq
    import os
    # IMPORTANTE: Importamos streamlit para leer los secrets si estamos en la nube
    try:
        import streamlit as st
    except ImportError:
        st = None

    print("🤖 Generando informe de operaciones con Llama-3...")

    # Buscamos la clave de forma segura: 
    # 1. En Secrets de Streamlit, 2. En variables de entorno, 3. Vacío si no hay nada
    api_key = ""
    if st and "GROQ_API_KEY" in st.secrets:
        api_key = st.secrets["GROQ_API_KEY"]
    else:
        api_key = os.environ.get("GROQ_API_KEY", "")

    if not api_key:
        return "⚠️ Error: GROQ_API_KEY no encontrada. No se puede generar el informe."

    client = Groq(api_key=api_key)

    horas_estres  = df_pred[df_pred['alerta'] == "🔴 ESTRÉS"]['hora'].tolist()
    horas_vigilar = df_pred[df_pred['alerta'] == "🟡 VIGILAR"]['hora'].tolist()
    consumo_total = df_pred['consumo_proyectado_m3'].sum()
    variacion_media = df_pred['variacion_pct'].mean()

    prompt = f"""
Eres el Jefe de Operaciones de Aguas de Alicante. Redacta el 'Informe Diario de Riesgo'.

DATOS DEL SECTOR {sector}:
- Volumen total proyectado hoy: {consumo_total:.2f} M3
- Horas en ESTRÉS confirmado: {horas_estres if horas_estres else 'Ninguna'}
- Horas en VIGILANCIA preventiva: {horas_vigilar if horas_vigilar else 'Ninguna'}
- Variación media proyectada: {variacion_media:+.1f}%

CONTEXTO SOCIOLÓGICO ACTUAL:
{contexto_texto}

INSTRUCCIONES:
1. Sé directo y gerencial.
2. Explica POR QUÉ esperamos esos consumos cruzando números con contexto sociológico.
3. Da recomendación táctica a operarios (válvulas, presión, zonas a vigilar).
4. Si no hay horas en estrés, confírmalo explícitamente para tranquilidad del equipo.
MÁXIMO 150 PALABRAS.
"""

    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error generando reporte con Groq: {e}"


def imprimir_reporte(df_pred: pd.DataFrame, sector: str, json_llm: dict) -> None:
    print("\n" + "=" * 65)
    print(f"  PREDICCIÓN HÍBRIDA — {sector}")
    print(f"  Factores LLM recibidos: {json_llm}")
    mult_medio = df_pred['multiplicador_llm'].mean()
    print(f"  Multiplicador LLM medio del día: x{mult_medio:.4f}")
    print("=" * 65)
    print(f"{'Hora':>5} {'Base (M³)':>12} {'Proyect. (M³)':>14} {'Var%':>8}  {'P85':>8}  Alerta")
    print("-" * 65)
    for _, row in df_pred.iterrows():
        print(f"{int(row['hora']):>5}h  "
              f"{row['consumo_base_m3']:>10.3f}   "
              f"{row['consumo_proyectado_m3']:>12.3f}   "
              f"{row['variacion_pct']:>+7.1f}%  "
              f"{row['p85_historico']:>8.3f}  {row['alerta']}")
    print("-" * 65)

    total_base  = df_pred["consumo_base_m3"].sum()
    total_final = df_pred["consumo_proyectado_m3"].sum()
    var_total   = ((total_final - total_base) / total_base) * 100

    conteo = df_pred["alerta"].value_counts()
    print(f"\n  TOTALES DEL DÍA")
    print(f"  Base XGBoost:      {total_base:.2f} M³")
    print(f"  Proyección final:  {total_final:.2f} M³  ({var_total:+.1f}%)")
    print(f"  🔴 ESTRÉS:         {conteo.get('🔴 ESTRÉS', 0)}/24 horas")
    print(f"  🟡 VIGILAR:        {conteo.get('🟡 VIGILAR', 0)}/24 horas")
    print(f"  🟢 NORMAL:         {conteo.get('🟢 NORMAL', 0)}/24 horas")
    print(f"  🔵 CAÍDA:          {conteo.get('🔵 CAÍDA', 0)}/24 horas")
    print("=" * 65)


# ==============================================================================
# EJECUCIÓN
# ==============================================================================

if __name__ == "__main__":
    RUTA_CSV          = "Dataset_24H_Mejorado_V2.csv"
    FECHA_PREDICCION  = "2026-04-06"

    df_historico, label_encoder = preparar_datos_ml(RUTA_CSV)
    modelo_xgb = entrenar_modelo(df_historico)

    print("\n🌐 Leyendo el mundo real mediante agente_llm.py...")
    factores_ia, contexto_dict = agente_llm.generar_factores_llm()

    es_ramadan   = contexto_dict['calendario']['es_ramadan']
    resumen_social = (
        f"Clima: {contexto_dict['clima']['resumen']}. "
        f"Fiestas: {contexto_dict['fiestas']['resumen']}. "
        f"Eventos: {contexto_dict['eventos']['resumen']}."
    )

    print("\n🚀 Calculando proyecciones para los 43 sectores...")
    sectores_unicos        = df_historico['SECTOR'].unique()
    todas_las_predicciones = []

    for sector in sectores_unicos:
        try:
            df_sector_pred = predecir_perfil_24h(
                modelo_xgb, df_historico, label_encoder, sector,
                FECHA_PREDICCION, factores_ia, es_ramadan
            )
            df_sector_pred.insert(0, 'sector', sector)
            todas_las_predicciones.append(df_sector_pred)
        except Exception as e:
            print(f"⚠️ Error calculando {sector}: {e}")

    df_maestro = pd.concat(todas_las_predicciones, ignore_index=True)

    # Diagnóstico rápido de distribución de alertas (útil para detectar futuros sesgos)
    print("\n📊 DISTRIBUCIÓN GLOBAL DE ALERTAS:")
    print(df_maestro["alerta"].value_counts().to_string())

    print("\n🤖 Generando informe global de operaciones con Llama-3...")
    sectores_en_estres = df_maestro[df_maestro['alerta'] == '🔴 ESTRÉS']['sector'].nunique()

    informe_global = generar_reporte_gerencial(
        df_maestro,
        sector=f"GLOBAL CIUDAD ({sectores_en_estres} sectores en estrés hoy)",
        contexto_texto=resumen_social
    )

    print("\n" + "📢" * 20)
    print("  INFORME GERENCIAL PARA STREAMLIT:")
    print("📢" * 20 + "\n")
    print(informe_global)

    with open("informe_global.txt", "w", encoding="utf-8") as f:
        f.write(informe_global)

    out_csv = f"prediccion_GLOBAL_ALICANTE_{FECHA_PREDICCION}.csv"
    df_maestro.to_csv(out_csv, index=False)
    print(f"\n💾 ¡Misión Cumplida! Archivo maestro exportado → {out_csv}")