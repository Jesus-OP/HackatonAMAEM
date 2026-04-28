"""
Conjunto.py — Motor Predictivo de Ensamble Híbrido (XGBoost + LLM)
===================================================================
Núcleo del Gemelo Digital de Predicción Hídrica. 
Implementa una arquitectura de fusión que combina un modelo puramente 
estadístico/inercial (XGBoost con TimeSeriesSplit) con un vector de 
tensores sociológicos generados en tiempo real (LLM Zero-Shot). 
Incluye un motor de enrutamiento espaciotemporal y reglas estrictas 
de dominio (Clipping y Ponderación) para garantizar la seguridad 
física de las predicciones en la red de abastecimiento.
"""

import os
import json
import warnings
from datetime import datetime, timedelta
import agente_llm
import joblib

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURACIÓN DE RUTAS
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# ==============================================================================
# HIPERPARÁMETROS Y LÍMITES OPERACIONALES (CLIPPING)
# ==============================================================================

# Límites de impacto por tensor individual
FACTOR_MIN = 0.70   
FACTOR_MAX = 1.50   

# Saturación del multiplicador global (Previene explosión combinatoria)
MULT_CAP_MIN = 0.60  # Límite inferior de desviación permitida (-40%)
MULT_CAP_MAX = 1.40  # Límite superior de desviación permitida (+40%)

# Umbrales de activación de alertas (Porcentaje de desviación respecto a la base)
UMBRAL_ESTRES = 15.0  
UMBRAL_CAIDA  = -15.0 


# ==============================================================================
# 1. INGENIERÍA DE CARACTERÍSTICAS (FEATURE ENGINEERING)
# ==============================================================================

def preparar_datos_ml(ruta_csv: str) -> tuple[pd.DataFrame, LabelEncoder]:
    """
    Ingesta del dataset procesado (Ground Truth) y construcción de variables 
    inerciales (Lags espaciotemporales) y cíclicas.
    """
    print("⚙️ Ingestando y vectorizando series temporales para entrenamiento...")
    df = pd.read_csv(ruta_csv)
    df["FECHA_HORA"] = pd.to_datetime(df["FECHA_HORA"])
    df = df.sort_values(["SECTOR", "FECHA_HORA"]).reset_index(drop=True)

    # Codificación de variables cíclicas temporales
    df["Hora"]           = df["FECHA_HORA"].dt.hour
    df["DiaSemana"]      = df["FECHA_HORA"].dt.dayofweek
    df["Mes"]            = df["FECHA_HORA"].dt.month
    df["Es_FinDeSemana"] = df["DiaSemana"].isin([5, 6]).astype(int)

    # Generación de retardos inerciales aislados por sector (Evita Data Leakage espacial)
    df["Lag_24h"]  = df.groupby("SECTOR")["CAUDAL_M3"].shift(24)
    df["Lag_168h"] = df.groupby("SECTOR")["CAUDAL_M3"].shift(168)

    le = LabelEncoder()
    df["SECTOR_ENC"] = le.fit_transform(df["SECTOR"])

    # Cálculo de percentiles históricos por sector y franja horaria para umbrales dinámicos
    df["p85_sector_hora"] = df.groupby(["SECTOR", "Hora"])["CAUDAL_M3"].transform(
        lambda x: x.quantile(0.85)
    )
    df["p15_sector_hora"] = df.groupby(["SECTOR", "Hora"])["CAUDAL_M3"].transform(
        lambda x: x.quantile(0.15)
    )

    cols_eliminar = [c for c in ["METODO"] if c in df.columns]
    df = df.drop(columns=cols_eliminar).dropna()

    print(f"  → Matriz resultante: {len(df):,} vectores | {df['SECTOR'].nunique()} sectores.")
    return df, le


# ==============================================================================
# 2. ENTRENAMIENTO ROBUSTO (TIME-SERIES SPLIT)
# ==============================================================================

FEATURES = ["Hora", "DiaSemana", "Mes", "Es_FinDeSemana",
            "Lag_24h", "Lag_168h", "SECTOR_ENC"]


def entrenar_modelo(df: pd.DataFrame) -> XGBRegressor:
    """
    Entrenamiento del modelo Gradient Boosting. Utiliza validación cruzada 
    temporal estricta para garantizar el respeto a la causalidad histórica.
    """
    print("\n🧠 Calibrando motor XGBoost (TimeSeriesSplit Cross-Validation)...")

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

    print(f"  ✅ Error Medio Absoluto (MAE) CV: {np.mean(maes):.3f} ± {np.std(maes):.3f} M³")

    # Reentrenamiento sobre el corpus completo para máxima capacidad de inferencia
    modelo_final = XGBRegressor(n_estimators=150, learning_rate=0.1,
                                max_depth=5, random_state=42, n_jobs=-1)
    modelo_final.fit(X, y)
    return modelo_final


# ==============================================================================
# 3. EXTRACCIÓN DE INERCIA EN TIEMPO REAL
# ==============================================================================

def _obtener_lag_real(df_sector: pd.DataFrame, timestamp: pd.Timestamp,
                      horas_atras: int) -> float:
    ts_objetivo = timestamp - timedelta(hours=horas_atras)
    fila = df_sector[df_sector["FECHA_HORA"] == ts_objetivo]

    if not fila.empty:
        return float(fila["CAUDAL_M3"].iloc[0])

    hora = ts_objetivo.hour
    media = df_sector[df_sector["Hora"] == hora]["CAUDAL_M3"].mean()
    warnings.warn(f"⚠️ Ruptura inercial en {ts_objetivo}. Imputando media histórica horaria.")
    return float(media)


# ==============================================================================
# 4. FUSIÓN HÍBRIDA (ENRUTAMIENTO Y PONDERACIÓN)
# ==============================================================================

# Reglas de enrutamiento espacial
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

# Matriz de Ponderación Dinámica (Relevancia de variables)
_PESOS_FACTOR = {
    "factor_global":              2.0,  
    "factor_calor_acumulado":     2.5,  
    "factor_eventos":             1.5,
    "factor_ocupacion_hotelera":  1.5,
    "factor_fin_de_semana":       1.0,
    "factor_franja_manana":       0.8,
    "factor_franja_tarde":        1.2,
    "factor_franja_noche":        0.6,
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
    try:
        v = float(valor_raw)
        if not np.isfinite(v):
            return 1.0
        return max(FACTOR_MIN, min(FACTOR_MAX, v))
    except (TypeError, ValueError):
        return 1.0


def aplicar_factores_llm(
    factores_ia: dict, zona: str, tramo: str, hora: int, es_ramadan: bool = False
) -> tuple[float, dict]:
    """
    Fusión matemática de tensores sociológicos mediante Media Ponderada.
    Integra reglas de dominio espacial y amortiguación de confianza del LLM.
    """
    if zona not in _FACTORES_POR_ZONA:
        raise ValueError(f"Enrutamiento espacial fallido. Zona '{zona}' desconocida.")

    activos = _FACTORES_POR_ZONA[zona] | _FACTORES_POR_TRAMO[tramo]
    if es_ramadan and zona == "ZONA_NORTE" and tramo == "noche":
        activos.add("factor_ramadan_nocturno")

    audit_trail = {}
    suma_ponderada = 0.0
    suma_pesos     = 0.0

    for nombre_factor in sorted(activos):
        valor = _sanitizar_factor(factores_ia.get(nombre_factor, 1.0), nombre_factor)
        peso  = _PESOS_FACTOR.get(nombre_factor, 1.0)

        # Sobrecarga dinámica de pesos por correlación espaciotemporal (ej. Playa + Calor + Tarde)
        if nombre_factor == "factor_calor_acumulado" and zona == "PLAYA_SAN_JUAN" and tramo in ("tarde", "noche"):
            peso *= 2.0
            audit_trail["_nota_calor"] = f"Ponderación agravada en PLAYA_SAN_JUAN."

        suma_ponderada += valor * peso
        suma_pesos     += peso
        audit_trail[nombre_factor] = round(valor, 4)

    multiplicador_raw = suma_ponderada / suma_pesos if suma_pesos > 0 else 1.0

    # Amortiguación basada en la confianza declarada por la capa Zero-Shot
    confianza = max(0.0, min(1.0, float(factores_ia.get("confianza", 1.0))))
    multiplicador_ajustado = 1.0 + (multiplicador_raw - 1.0) * confianza

    # Saturación de seguridad (Clipping general)
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
# 5. INFERENCIA Y PROYECCIÓN 24H
# ==============================================================================

def _calcular_alerta(consumo_proyectado: float, consumo_base: float,
                     variacion_pct: float, p85: float, p15: float) -> str:
    """Clasificación del nivel de estrés hídrico basada en anomalía vs. histórico."""
    sobre_percentil_alto = consumo_proyectado > p85
    bajo_percentil_bajo  = consumo_proyectado < p15

    if variacion_pct > UMBRAL_ESTRES and sobre_percentil_alto:
        return "🔴 ESTRÉS"
    elif variacion_pct < UMBRAL_CAIDA and bajo_percentil_bajo:
        return "🔵 CAÍDA"
    elif variacion_pct > UMBRAL_ESTRES:
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
        raise ValueError(f"Fallo de inferencia: Sector '{sector_objetivo}' aislado.")

    sector_enc = int(le.transform([sector_objetivo])[0])
    zona       = asignar_macrozona(sector_objetivo)
    registros  = []

    for hora in range(24):
        ts    = fecha_base + timedelta(hours=hora)
        tramo = hora_a_tramo(hora)

        lag_24h  = _obtener_lag_real(df_sector, ts, 24)
        lag_168h = _obtener_lag_real(df_sector, ts, 168)

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
        variacion_pct      = (multiplicador - 1.0) * 100 

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
# 6. EXPLICABILIDAD GENERATIVA (XAI)
# ==============================================================================

def generar_reporte_gerencial(df_pred, sector, contexto_texto):
    """Traducción de métricas operativas a lenguaje natural mediante LLM."""
    from groq import Groq
    
    try:
        import streamlit as st
    except ImportError:
        st = None

    print("🤖 Sintetizando informe operativo (XAI)...")

    api_key = ""
    if st and "GROQ_API_KEY" in st.secrets:
        api_key = st.secrets["GROQ_API_KEY"]
    else:
        api_key = os.environ.get("GROQ_API_KEY", "")

    if not api_key:
        return "⚠️ Alerta de Sistema: API Key ausente. Generación de informe operativo abortada."

    client = Groq(api_key=api_key)

    horas_estres  = df_pred[df_pred['alerta'] == "🔴 ESTRÉS"]['hora'].tolist()
    horas_vigilar = df_pred[df_pred['alerta'] == "🟡 VIGILAR"]['hora'].tolist()
    consumo_total = df_pred['consumo_proyectado_m3'].sum()
    variacion_media = df_pred['variacion_pct'].mean()

    prompt = f"""
Actúa como Analista Principal de la red de abastecimiento. Genera el 'Informe de Riesgo Operativo'.

MÉTRICAS:
- Área: {sector}
- Volumen proyectado: {consumo_total:.2f} M3
- Horas Críticas (Estrés): {horas_estres if horas_estres else 'Ninguna'}
- Horas de Vigilancia: {horas_vigilar if horas_vigilar else 'Ninguna'}
- Tasa de variación: {variacion_media:+.1f}%

CONTEXTO DE INFERENCIA:
{contexto_texto}

REGLAS DE REDACCIÓN:
1. Tono estrictamente analítico, conciso y técnico.
2. Relaciona la desviación del caudal esperado con los vectores sociológicos detectados.
3. Provee una recomendación táctica de ajuste en la infraestructura.
4. Límite estricto: 150 palabras.
"""

    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.2,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error en orquestación de informe: {e}"


# ==============================================================================
# PIPELINE DE EJECUCIÓN 
# ==============================================================================

if __name__ == "__main__":
    
    # Garantizar la existencia del directorio de salida
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    RUTA_CSV = os.path.join(PROCESSED_DIR, "Dataset_24H_Mejorado_V2.csv")
    FECHA_PREDICCION = datetime.now().strftime("%Y-%m-%d")

    df_historico, label_encoder = preparar_datos_ml(RUTA_CSV)
    
    # Ruta donde se guardará/leerá el modelo congelado
    RUTA_MODELO = os.path.join(BASE_DIR, "modelos", "modelo_xgb_final.joblib")
    
    if os.path.exists(RUTA_MODELO):
        print("\n⚡ Cargando modelo XGBoost pre-entrenado (Modo Demo Ultrarrápido)...")
        modelo_xgb = joblib.load(RUTA_MODELO)
    else:
        print("\n🧠 Modelo no encontrado. Entrenando por primera vez...")
        os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
        modelo_xgb = entrenar_modelo(df_historico)
        joblib.dump(modelo_xgb, RUTA_MODELO)
        print(f"💾 Modelo guardado en {RUTA_MODELO} para futuras ejecuciones.")

    print("\n🌐 Ejecutando Capa Sensorial OSINT (agente_llm)...")
    factores_ia, contexto_dict = agente_llm.generar_factores_llm()

    es_ramadan = contexto_dict['calendario']['es_ramadan']
    resumen_social = (
        f"Clima: {contexto_dict['clima']['resumen']}. "
        f"Fiestas: {contexto_dict['fiestas']['resumen']}. "
        f"Eventos: {contexto_dict['eventos']['resumen']}."
    )

    print(f"\n🚀 Procesando inferencia espacial iterativa para {df_historico['SECTOR'].nunique()} sectores...")
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
            print(f"⚠️ Omisión en {sector}: {e}")

    df_maestro = pd.concat(todas_las_predicciones, ignore_index=True)

    print("\n📊 DISTRIBUCIÓN GLOBAL DE CLASIFICACIÓN DE ALERTA:")
    print(df_maestro["alerta"].value_counts().to_string())

    print("\n🤖 Consolidando Informe Operativo Global...")
    sectores_en_estres = df_maestro[df_maestro['alerta'] == '🔴 ESTRÉS']['sector'].nunique()

    informe_global = generar_reporte_gerencial(
        df_maestro,
        sector=f"GLOBAL CIUDAD ({sectores_en_estres} sectores en estrés crítico)",
        contexto_texto=resumen_social
    )

    print("\n" + "=" * 65)
    print("  INFORME EJECUTIVO:")
    print("=" * 65 + "\n")
    print(informe_global)

    ruta_informe = os.path.join(OUTPUT_DIR, "informe_global.txt")
    with open(ruta_informe, "w", encoding="utf-8") as f:
        f.write(informe_global)

    out_csv = os.path.join(OUTPUT_DIR, f"prediccion_GLOBAL_ALICANTE_{FECHA_PREDICCION}.csv")
    df_maestro.to_csv(out_csv, index=False)
    print(f"\n💾 Pipeline completado con éxito. Resultados exportados a:\n -> {out_csv}")