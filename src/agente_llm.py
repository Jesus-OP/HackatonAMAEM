"""
agente_llm.py — Capa Sensorial de Contexto Sociológico (Zero-Shot Feature Extraction)
=====================================================================================
Módulo encargado de actuar como transductor de datos no estructurados en tiempo real.
Mediante llamadas a APIs públicas (OSINT) y el uso de Inteligencia Artificial Generativa 
(Llama-3 70B vía Groq), el sistema procesa el entorno social, climático y urbano de la ciudad,
exportando un vector de tensores numéricos (factores de impacto) que el modelo de Machine 
Learning utilizará para ajustar la predicción inercial de la red hídrica.
"""

import requests
import xml.etree.ElementTree as ET
from groq import Groq
from datetime import datetime, date
from bs4 import BeautifulSoup
import json
import os
import pandas as pd

# =============================================================================
# ⚙️ CONFIGURACIÓN DE RUTAS Y ENTORNO
# =============================================================================

# El script se encuentra en /src, el BASE_DIR es la raíz del proyecto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUTA_CSV_EVENTOS = os.path.join(BASE_DIR, "data", "raw", "aguas_corregido_v2_Sheet1_.csv")
RUTA_OUTPUT_JSON = os.path.join(BASE_DIR, "outputs", "factores_hoy.json")

def cargar_config(key_name):
    # 1. Resolución mediante Streamlit Secrets (Entorno Cloud)
    try:
        import streamlit as st
        if key_name in st.secrets:
            return st.secrets[key_name]
    except ImportError:
        pass
    
    # 2. Resolución mediante Variables de Entorno (Entorno Local/Docker)
    return os.getenv(key_name, "")

API_KEY          = cargar_config("GROQ_API_KEY")
TICKETMASTER_KEY = cargar_config("TICKETMASTER_KEY")
MODELO           = "llama-3.3-70b-versatile"

if API_KEY:
    cliente_ai = Groq(api_key=API_KEY)
else:
    print("⚠️ [FALLBACK] GROQ_API_KEY no detectada. El sistema operará en modo determinista (priors matemáticos).")


# =============================================================================
# 1. MÓDULO CLIMÁTICO
# =============================================================================

def obtener_clima_alicante() -> dict:
    try:
        url = (
            "https://api.open-meteo.com/v1/forecast"
            "?latitude=38.3452&longitude=-0.4815"
            "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum"
            "&hourly=temperature_2m,relativehumidity_2m,windspeed_10m"
            "&timezone=Europe/Madrid&forecast_days=2"
        )
        r         = requests.get(url, timeout=5).json()
        temps_h   = r['hourly']['temperature_2m'][24:48]
        humedad_h = r['hourly']['relativehumidity_2m'][24:48]
        viento_h  = r['hourly']['windspeed_10m'][24:48]
        temp_max  = r['daily']['temperature_2m_max'][1]
        temp_min  = r['daily']['temperature_2m_min'][1]
        lluvia    = r['daily']['precipitation_sum'][1]
        hora_pico = temps_h.index(max(temps_h))
        horas_28  = sum(1 for t in temps_h if t > 28)
        horas_32  = sum(1 for t in temps_h if t > 32)
        st = _sensacion_termica(
            sum(temps_h[14:21])/7, sum(humedad_h[14:21])/7, sum(viento_h[14:21])/7
        )
        return {
            "temp_max": temp_max, "temp_min": temp_min, "lluvia_mm": lluvia,
            "hora_pico_calor": hora_pico, "horas_sobre_28c": horas_28,
            "horas_sobre_32c": horas_32, "sensacion_tarde": round(st, 1),
            "resumen": (
                f"{_resumir_clima(temp_max, lluvia)} | Pico {hora_pico:02d}h | "
                f"{horas_28}h >28C | Sensacion tarde {st:.0f}C"
            )
        }
    except Exception as e:
        print(f"  [WARN] Open-Meteo API: {e}")
        return {"temp_max":20,"temp_min":14,"lluvia_mm":0,"hora_pico_calor":15,
                "horas_sobre_28c":0,"horas_sobre_32c":0,"sensacion_tarde":20,
                "resumen":"Climatología base estimada."}

def _resumir_clima(temp, lluvia):
    if temp > 35:   return f"Ola de calor extrema: {temp}C"
    if temp > 30:   return f"Alerta térmica alta: {temp}C"
    if lluvia > 10: return f"Precipitación intensa: {lluvia}mm, {temp}C"
    if lluvia > 0:  return f"Precipitación leve: {lluvia}mm, {temp}C"
    return f"Despejado: {temp}C"

def _sensacion_termica(temp, humedad, viento):
    if temp < 27: return temp - viento * 0.1
    return (-8.784695 + 1.61139411*temp + 2.338549*humedad
            - 0.14611605*temp*humedad - 0.01230809*temp**2
            - 0.01642482*humedad**2 + 0.00221173*temp**2*humedad
            + 0.00072546*temp*humedad**2 - 0.00000358*temp**2*humedad**2)


# =============================================================================
# 2. MÓDULO CALENDARIO Y CICLOS SOCIALES
# =============================================================================

def obtener_calendario() -> dict:
    hoy = datetime.now()
    res = {
        "fecha": hoy.strftime("%Y-%m-%d"), "dia_semana": hoy.strftime("%A"),
        "dia_numero": hoy.weekday(), "es_fin_semana": hoy.weekday() >= 5,
        "es_festivo": False, "nombre_festivo": None,
        "es_ramadan": _es_ramadan(hoy),
        "escolar":    _estado_escolar(hoy),
        "perfil_dia": _perfil_dia(hoy),
    }
    try:
        festivos = requests.get(
            f"https://date.nager.at/api/v3/PublicHolidays/{hoy.year}/ES", timeout=5
        ).json()
        f = next((f for f in festivos if f['date'] == hoy.strftime("%Y-%m-%d")), None)
        if f:
            res['es_festivo'] = True
            res['nombre_festivo'] = f.get('localName', f.get('name'))
    except Exception as e:
        print(f"  [WARN] Festivos API: {e}")
    return res

def _es_ramadan(fecha):
    rangos = {2025:((3,1),(3,30)), 2026:((2,18),(3,19)), 2027:((2,8),(3,9))}
    r = rangos.get(fecha.year)
    if not r: return False
    return datetime(fecha.year,r[0][0],r[0][1]) <= fecha <= datetime(fecha.year,r[1][0],r[1][1])

def _estado_escolar(fecha):
    m, d = fecha.month, fecha.day
    vac, per = False, "Ciclo lectivo ordinario"
    if (m==6 and d>=15) or m in [7,8] or (m==9 and d<=10): vac, per = True, "Periodo estival escolar"
    elif (m==12 and d>=23) or (m==1 and d<=7):              vac, per = True, "Receso invernal"
    elif _es_semana_santa(fecha):                            vac, per = True, "Semana Santa"
    patron = ("Demanda matutina retrasada (09:00-10:00)." if vac else
              "Pico matutino estándar (07:00-08:30)." if fecha.weekday()<5 else "Patrón de fin de semana.")
    return {"es_vacaciones": vac, "periodo": per, "patron": patron}

def _es_semana_santa(fecha):
    a=fecha.year%19; b,c=fecha.year//100,fecha.year%100
    d,e=b//4,b%4; f=(b+8)//25; g=(b-f+1)//3
    h=(19*a+b-d-g+15)%30; i,k=c//4,c%4
    l=(32+2*e+2*i-h-k)%7; m=(a+11*h+22*l)//451
    mp=(h+l-7*m+114)//31; dp=((h+l-7*m+114)%31)+1
    pascua=date(fecha.year,mp,dp)
    dr=dp-7
    ramos=date(fecha.year,mp,dr) if dr>0 else date(fecha.year,mp-1,30+dr)
    return ramos<=fecha.date()<=pascua

def _perfil_dia(fecha):
    p={
        0:{"nombre":"Lunes",    "f_man":1.10,"f_tar":0.95,"f_noc":0.90,"nota":"Ajuste inercial post-fin de semana."},
        1:{"nombre":"Martes",   "f_man":1.00,"f_tar":1.00,"f_noc":0.95,"nota":"Día laborable estándar."},
        2:{"nombre":"Miercoles","f_man":1.00,"f_tar":1.00,"f_noc":0.95,"nota":"Día laborable estándar."},
        3:{"nombre":"Jueves",   "f_man":1.00,"f_tar":1.02,"f_noc":1.02,"nota":"Incremento leve ocio nocturno."},
        4:{"nombre":"Viernes",  "f_man":1.00,"f_tar":1.05,"f_noc":1.10,"nota":"Tarde-noche con alta movilidad."},
        5:{"nombre":"Sabado",   "f_man":0.90,"f_tar":1.10,"f_noc":1.15,"nota":"Actividad residencial y ocio intenso."},
        6:{"nombre":"Domingo",  "f_man":0.85,"f_tar":1.15,"f_noc":0.90,"nota":"Pico de demanda en mediodía."},
    }[fecha.weekday()]
    return {"nombre":p["nombre"],"f_manana":p["f_man"],"f_tarde":p["f_tar"],
            "f_noche":p["f_noc"],"nota":p["nota"]}


# =============================================================================
# 3. FESTIVIDADES LOCALES Y REGIONALES
# =============================================================================

FIESTAS_ALICANTE = [
    (6,20,24,"Hogueras de San Juan",1.30,"Afluencia masiva regional. Pico hídrico nocturno."),
    (6,23,23,"Noche de la Crema",   1.40,"Pico máximo de turismo. Estrés en red de abastecimiento."),
    (6,17,19,"Pre-Hogueras",        1.10,"Incremento progresivo de ocupación."),
    (12,25,26,"Navidad",            0.80,"Concentración residencial. Actividad comercial nula."),
    (12,31,31,"Nochevieja",         1.20,"Concentración urbana masiva zona centro."),
    (1,1,1,"Ano Nuevo",             0.75,"Actividad hídrica mínima (valle generalizado)."),
    (3,19,19,"San Jose",            0.95,"Festivo autonómico."),
    (8,1,8,"Moros y Cristianos",    1.10,"Impacto localizado en zona metropolitana norte."),
]

def obtener_fiestas_alicante() -> dict:
    hoy = datetime.now()
    mes, dia = hoy.month, hoy.day
    fh = fm = None
    for (fm_,fi,ff,n,fac,d) in FIESTAS_ALICANTE:
        if fm_==mes and fi<=dia<=ff:    fh = {"nombre":n,"factor":fac,"detalle":d}
        if fm_==mes and fi<=(dia+1)<=ff: fm = {"nombre":n,"factor":fac,"detalle":d}
    hogueras = (mes==6 and 17<=dia<=24)
    return {
        "fiesta_hoy": fh, "fiesta_manana": fm, "es_semana_hogueras": hogueras,
        "resumen": (
            f"FIESTA ACTIVA HOY: {fh['nombre']} (Multiplicador base: x{fh['factor']}) — {fh['detalle']}" if fh else
            f"FIESTA PROYECTADA MAÑANA: {fm['nombre']}" if fm else
            "SEMANA HOGUERAS: Actividad festiva en progreso." if hogueras else
            "Sin festividades locales detectadas."
        )
    }


# =============================================================================
# 4. CALIDAD DEL AIRE
# =============================================================================

def obtener_calidad_aire() -> dict:
    try:
        r = requests.get(
            "https://air-quality-api.open-meteo.com/v1/air-quality"
            "?latitude=38.3452&longitude=-0.4815&current=dust,pm10,pm2_5&timezone=Europe/Madrid",
            timeout=5
        ).json()
        polvo = r['current']['dust']
        return {"polvo_ug_m3":polvo,"alerta_calima":polvo>50,
                "resumen":f"ALERTA CALIMA: {polvo} ug/m3 detectados." if polvo>50 else "Parámetros de calidad del aire normales."}
    except Exception as e:
        print(f"  [WARN] Air Quality API: {e}")
        return {"polvo_ug_m3":0,"alerta_calima":False,"resumen":"Parámetros estimados (normalidad)."}


# =============================================================================
# 5. MÓDULO MARÍTIMO (Cruceros en Puerto de Alicante)
# =============================================================================

def obtener_cruceros_alicante() -> dict:
    try:
        hoy = datetime.now()
        url = f"https://www.cruisewatch.com/port/alicante-spain/schedule/{hoy.year}/{hoy.month:02d}"
        r   = requests.get(url, timeout=8, headers={"User-Agent":"Mozilla/5.0"})
        soup = BeautifulSoup(r.text, "html.parser")
        hoy_str = hoy.strftime("%d")
        cruceros_hoy = []
        for fila in soup.find_all("tr"):
            celdas = fila.find_all("td")
            if len(celdas) >= 2 and hoy_str in fila.get_text():
                nombre = celdas[1].get_text(strip=True)
                cruceros_hoy.append({"barco":nombre,"pasajeros_est":_est_pax(nombre)})
        total = sum(c["pasajeros_est"] for c in cruceros_hoy)
        
        # Calibración estricta de impacto en el suministro del puerto
        factor = round(min(1.0 + (total / 15000) * 0.25, 1.25), 2)
        return {
            "cruceros_hoy": cruceros_hoy, "num_cruceros": len(cruceros_hoy),
            "pasajeros_totales": total, "factor_impacto": factor,
            "resumen": (
                f"OPERATIVA PORTUARIA: {len(cruceros_hoy)} embarcaciones detectadas. "
                f"Volumen estimado de pasajeros flotantes: ~{total:,}."
                if cruceros_hoy else "Ausencia de tráfico de cruceros detectado en el puerto."
            )
        }
    except Exception as e:
        print(f"  [WARN] Módulo Marítimo: {e}")
        return {"cruceros_hoy":[],"num_cruceros":0,"pasajeros_totales":0,
                "factor_impacto":1.0,"resumen":"Sin confirmación de tráfico portuario turístico."}

def _est_pax(nombre):
    n = nombre.lower()
    if any(x in n for x in ["symphony","wonder","icon","oasis","allure"]): return 5000
    if any(x in n for x in ["msc","costa","carnival","celebrity"]):         return 3000
    return 1500


# =============================================================================
# 6. MÓDULO TRÁFICO AÉREO (OpenSky Network)
# =============================================================================

def obtener_vuelos_alicante() -> dict:
    try:
        r = requests.get(
            "https://opensky-network.org/api/states/all"
            "?lamin=38.27&lomin=-0.59&lamax=38.32&lomax=-0.53",
            timeout=8
        ).json()
        estados = r.get("states", []) or []
        aterrizando  = [s for s in estados if s[11] and s[11]<-2 and s[5] and s[5]<1500]
        aproximacion = [s for s in estados if s[5] and s[5]<3000 and s[9] and s[9]<300]
        mes = datetime.now().month
        
        # Factor estacional ponderado
        factor_est = 1.20 if mes in [7,8] else 1.10 if mes in [6,9] else 1.0
        return {
            "vuelos_aterrizando": len(aterrizando),
            "vuelos_aproximacion": len(aproximacion),
            "pasajeros_est_hora": len(aproximacion)*150,
            "factor_estacional": factor_est,
            "resumen": (
                f"TRÁFICO AÉREO (ALC): {len(aterrizando)} arribos en tiempo real | {len(aproximacion)} en vector de aproximación. "
                f"(Flujo estimado: ~{len(aproximacion)*150} pax/h) | Multiplicador estacional: x{factor_est}"
            )
        }
    except Exception as e:
        print(f"  [WARN] OpenSky API: {e}")
        mes = datetime.now().month
        factor = 1.20 if mes in [7,8] else 1.10 if mes in [6,9] else 1.0
        return {"vuelos_aterrizando":0,"vuelos_aproximacion":0,"pasajeros_est_hora":0,
                "factor_estacional":factor,"resumen":f"Multiplicador estacional aplicado: x{factor} (Mes: {mes})"}


# =============================================================================
# 7. EVENTOS MULTITUDINARIOS (Integración AMAEM + Ticketmaster)
# =============================================================================

def cargar_eventos_csv_hoy() -> list[dict]:
    """Ingesta del registro oficial de eventos AMAEM."""
    eventos_hoy = []
    if not os.path.exists(RUTA_CSV_EVENTOS):
        print(f"  [WARN] Repositorio de eventos no localizado en: {RUTA_CSV_EVENTOS}")
        return []

    try:
        df = pd.read_csv(RUTA_CSV_EVENTOS, sep=';', encoding='latin1')
        df['FECHA_INICIO'] = pd.to_datetime(df['FECHA_INICIO'], format='mixed', dayfirst=False)
        df['FECHA_FIN']    = pd.to_datetime(df['FECHA_FIN'],    format='mixed', dayfirst=False)

        hoy = datetime.now().date()

        mask = (df['FECHA_INICIO'].dt.date <= hoy) & (df['FECHA_FIN'].dt.date >= hoy)
        df_hoy = df[mask].copy()

        for _, row in df_hoy.iterrows():
            eventos_hoy.append({
                "nombre":  row['TIPO_EVENTO'],
                "venue":   str(row['BARRIO_AFECTADO']),
                "hora":    str(row['HORA']),
                "aforo":   _aforo_desde_impacto(row['IMPACTO']),
                "zona":    _zona_desde_barrio(str(row['BARRIO_AFECTADO'])),
                "impacto": int(row['IMPACTO']),
                "fuente":  "Registro AMAEM",
            })

        if eventos_hoy:
            print(f"  [DATA] Registrados {len(eventos_hoy)} evento(s) confirmados para la jornada de hoy.")
        else:
            print("  [DATA] Ausencia de eventos programados para la jornada de hoy.")

    except Exception as e:
        print(f"  [WARN] Error de ingesta CSV Eventos: {e}")

    return eventos_hoy

def _aforo_desde_impacto(impacto):
    tabla = {1: 500, 2: 2000, 3: 5000, 4: 10000, 5: 50000}
    return tabla.get(int(impacto), 1000)

def _zona_desde_barrio(barrio_str: str) -> str:
    b = barrio_str.lower()
    if any(x in b for x in ["playa", "cabo", "postiguet", "muchavista"]): return "playa"
    if any(x in b for x in ["remedio", "carolinas", "campoamor", "ciudad deportiva",
                              "colonia requena", "altozano"]): return "norte"
    if b == "todos": return "global"
    return "centro"

def obtener_eventos_ticketmaster() -> dict:
    """Enriquecimiento de eventos mediante API externa, con fallback al registro oficial."""
    if TICKETMASTER_KEY:
        try:
            hoy = datetime.now()
            r = requests.get(
                "https://app.ticketmaster.com/discovery/v2/events.json",
                params={
                    "apikey": TICKETMASTER_KEY, "city": "Alicante", "countryCode": "ES",
                    "startDateTime": hoy.strftime("%Y-%m-%dT00:00:00Z"),
                    "endDateTime":   hoy.strftime("%Y-%m-%dT23:59:59Z"), "size": 10,
                }, timeout=8
            ).json()
            raw = r.get("_embedded", {}).get("events", [])
            eventos = []
            for ev in raw:
                nombre = ev.get("name","Evento")
                venue  = ev.get("_embedded",{}).get("venues",[{}])[0].get("name","Recinto")
                hora   = ev.get("dates",{}).get("start",{}).get("localTime","20:00")
                aforo  = _est_aforo_tm(nombre, venue)
                zona   = _zona_venue_tm(venue)
                eventos.append({"nombre":nombre,"venue":venue,"hora":hora,
                                 "aforo":aforo,"zona":zona,"fuente":"Ticketmaster"})
                
            eventos_csv = cargar_eventos_csv_hoy()
            eventos_csv_no_dup = [e for e in eventos_csv
                                  if not any(e['nombre'].lower() in ev['nombre'].lower()
                                             for ev in eventos)]
            eventos.extend(eventos_csv_no_dup)
            factor = round(min(1.0 + sum(e['aforo'] for e in eventos)/20000*0.10, 1.30), 2)
            return {
                "eventos_hoy": eventos, "num_eventos": len(eventos), "factor_evento": factor,
                "resumen": (
                    " | ".join([f"{e['nombre']} ({e['hora']}, Aforo Est.: ~{e['aforo']}, Zona: {e['zona']})"
                                for e in eventos])
                    if eventos else "Registro de eventos vacío en el municipio."
                )
            }
        except Exception as e:
            print(f"  [WARN] Ticketmaster API: {e} — Transicionando a fuente de datos estática.")

    eventos_csv = cargar_eventos_csv_hoy()
    factor = round(min(1.0 + sum(e['aforo'] for e in eventos_csv)/20000*0.10, 1.25), 2)
    return {
        "eventos_hoy": eventos_csv, "num_eventos": len(eventos_csv), "factor_evento": factor,
        "resumen": (
            " | ".join([f"{e['nombre']} ({e['hora']}, Zona: {e['zona']}, Índice Impacto: {e['impacto']}/5)"
                        for e in eventos_csv])
            if eventos_csv else "Registro de eventos vacío en el municipio."
        )
    }

def _est_aforo_tm(nombre, venue):
    nv = (nombre+venue).lower()
    if "estadio" in nv or "hercules" in nv: return 20000
    if "auditorio" in nv or "palacio" in nv: return 3000
    if "plaza" in nv and "toros" in nv: return 10000
    if "festival" in nv or "concert" in nv: return 5000
    return 1000

def _zona_venue_tm(venue):
    v = venue.lower()
    if any(x in v for x in ["rico perez","estadio","hercules"]): return "norte"
    if any(x in v for x in ["playa","san juan","postiguet"]):    return "playa"
    return "centro"


# =============================================================================
# 8. MÓDULO MOVILIDAD (MITMA)
# =============================================================================

def obtener_movilidad_mitma() -> dict:
    try:
        url = "https://servicios.fomento.gob.es/BDOTC/api/v1/movilidad/provincia/03"
        r   = requests.get(url, timeout=8,
                           headers={"Accept":"application/json","User-Agent":"Mozilla/5.0"})
        datos = r.json()
        indice = datos.get("indice_movilidad", 100)
        variacion = datos.get("variacion_mensual", 0)
        factor = round(indice / 100, 2)
        return {
            "indice_movilidad": indice, "variacion_mensual": variacion,
            "factor_movilidad": factor,
            "resumen": f"Índice de Movilidad Activa (MITMA): {indice} ({variacion:+.1f}% intermensual). Multiplicador x{factor}"
        }
    except Exception as e:
        print(f"  [WARN] MITMA API: {e} — Utilizando calibración estacional.")
        return _movilidad_estacional()

def _movilidad_estacional() -> dict:
    mes = datetime.now().month
    indice_por_mes = {
        1:88, 2:90, 3:95, 4:100, 5:102,
        6:108, 7:130, 8:140, 9:115, 10:100,
        11:92, 12:95
    }
    indice = indice_por_mes.get(mes, 100)
    factor = round(indice / 100, 2)
    return {
        "indice_movilidad": indice, "variacion_mensual": 0, "factor_movilidad": factor,
        "resumen": f"Estimación inercial de movilidad: {indice}/100. Multiplicador ponderado: x{factor}."
    }


# =============================================================================
# 9. INFRAESTRUCTURAS Y OBRAS
# =============================================================================

def obtener_obras_alicante() -> dict:
    try:
        url = (
            "https://datos.alicante.es/api/3/action/datastore_search"
            "?resource_id=licencias-obras-mayores&limit=20"
        )
        r    = requests.get(url, timeout=8, headers={"User-Agent":"Mozilla/5.0"}).json()
        obras = r.get("result", {}).get("records", [])
        obras_activas = []
        for obra in obras:
            estado = str(obra.get("estado","")).lower()
            if "activ" in estado or "en curso" in estado or "concedid" in estado:
                zona = _detectar_zona_obra(str(obra.get("direccion","") + obra.get("zona","")))
                obras_activas.append({
                    "descripcion": obra.get("descripcion", "Intervención infraestructural"),
                    "direccion":   obra.get("direccion", ""),
                    "zona":        zona, "estado": obra.get("estado",""),
                })
        factor = round(min(1.0 + len(obras_activas)*0.01, 1.10), 2)
        return {
            "obras_activas": obras_activas[:5], "total_obras": len(obras_activas),
            "factor_obras": factor,
            "resumen": (
                f"INTERVENCIONES VÍA PÚBLICA: {len(obras_activas)} obras mayores detectadas. Factor de riesgo: x{factor}."
                if obras_activas else "Ausencia de alteraciones reportadas en vía pública."
            )
        }
    except Exception as e:
        print(f"  [WARN] OpenData Obras: {e}")
        return {"obras_activas":[],"total_obras":0,"factor_obras":1.0,
                "resumen":"Telemetría de obras temporalmente no disponible."}

def _detectar_zona_obra(texto: str) -> str:
    t = texto.lower()
    if any(x in t for x in ["playa","san juan","cabo","postiguet"]): return "playa"
    if any(x in t for x in ["norte","carolinas","remedio","florida"]): return "norte"
    return "centro"


# =============================================================================
# 10. MÓDULO TURÍSTICO (INE)
# =============================================================================

def obtener_ocupacion_hotelera_ine() -> dict:
    try:
        url = "https://servicios.ine.es/wstempus/js/ES/DATOS_SERIE/IH9009?nult=13"
        r   = requests.get(url, timeout=8).json()
        datos_serie = r.get("Data", [])
        if not datos_serie: raise ValueError("Respuesta estructuralmente vacía.")
        ultimo      = datos_serie[-1]
        anterior    = datos_serie[-2] if len(datos_serie) > 1 else ultimo
        mismo_mes_anterior_anio = datos_serie[-13] if len(datos_serie) >= 13 else ultimo
        viajeros_ultimo   = ultimo.get("Valor", 0)
        viajeros_anterior = anterior.get("Valor", 1)
        viajeros_anio_ant = mismo_mes_anterior_anio.get("Valor", 1)
        variacion_mensual = round((viajeros_ultimo / viajeros_anterior - 1) * 100, 1)
        variacion_anual   = round((viajeros_ultimo / viajeros_anio_ant - 1) * 100, 1)
        media_anual = sum(d.get("Valor",0) for d in datos_serie[:-1]) / max(len(datos_serie)-1, 1)
        factor      = round(viajeros_ultimo / media_anual, 2) if media_anual > 0 else 1.0
        factor      = max(0.7, min(factor, 1.50))
        return {
            "viajeros_ultimo_mes": int(viajeros_ultimo),
            "variacion_mensual_pct": variacion_mensual,
            "variacion_anual_pct": variacion_anual,
            "factor_ocupacion": factor,
            "periodo": ultimo.get("NombrePeriodo", ""),
            "resumen": (
                f"REPORTE INE (EOH): {int(viajeros_ultimo):,} pernóctas registradas "
                f"({variacion_mensual:+.1f}% vs periodo anterior). Multiplicador x{factor}"
            )
        }
    except Exception as e:
        print(f"  [WARN] INE EOH: {e} — Utilizando calibración estacional.")
        return _ocupacion_estacional()

def _ocupacion_estacional() -> dict:
    mes = datetime.now().month
    ocupacion_mes = {
        1:60, 2:63, 3:70, 4:80, 5:85,
        6:92, 7:115, 8:125, 9:105, 10:85,
        11:65, 12:70
    }
    ocup   = ocupacion_mes.get(mes, 80)
    factor = round(ocup / 100, 2)
    return {
        "viajeros_ultimo_mes": 0, "variacion_mensual_pct": 0, "variacion_anual_pct": 0,
        "factor_ocupacion": factor, "periodo": f"Inferencia mes {mes}",
        "resumen": f"Carga hotelera estimada: {ocup}/100. Multiplicador ponderado: x{factor}."
    }


# =============================================================================
# 11. INTELIGENCIA DE FUENTES ABIERTAS (OSINT)
# =============================================================================

def escuchar_redes_sociales() -> str:
    fragmentos = []
    for nombre, url in [
        ("Diario Informacion", "https://www.diarioinformacion.com/elementosInt/rss/1"),
        ("Levante EMV",        "https://www.levante-emv.com/rss/section/portada"),
    ]:
        try:
            r = requests.get(url, timeout=5)
            r.encoding = 'utf-8'
            c = r.content
            if c.startswith(b'\xef\xbb\xbf'): c = c[3:]
            root = ET.fromstring(c.replace(b'\x00',b''))
            titulares = [i.find('title').text for i in root.findall('./channel/item')[:3]
                         if i.find('title') is not None and i.find('title').text]
            if titulares:
                fragmentos.append(f"PRENSA LOCAL (RSS): {' | '.join(titulares)}")
                break
        except Exception as e:
            print(f"  [WARN] RSS {nombre}: {e}")

    if not any('RSS' in f for f in fragmentos):
        fragmentos.append("PRENSA LOCAL: Flujo de datos temporalmente inactivo.")

    try:
        r = requests.get("https://www.reddit.com/r/alicante/new.json?limit=10",
                         headers={'User-Agent':'AquaAlert-Bot/1.0'}, timeout=5).json()
        kw = ['agua','calor','corte','averia','inundacion','sequia','lluvia','heat','water']
        rel = [p['data']['title'] for p in r['data']['children']
               if any(w in p['data']['title'].lower() for w in kw)]
        fragmentos.append(f"FOROS COMUNITARIOS (Reddit): {' | '.join(rel[:3])}" if rel else "FOROS COMUNITARIOS: Ausencia de palabras clave críticas.")
    except Exception as e:
        print(f"  [WARN] API Comunitaria: {e}")
        fragmentos.append("FOROS COMUNITARIOS: Estado nominal. Sin anomalías.")

    return "\n".join(fragmentos)


# =============================================================================
# TOPOLOGÍA URBANA
# =============================================================================

PERFIL_CIUDAD = """
ZONA NORTE (Sectores Virgen del Remedio, Carolinas):
  - Demografía: 85.000 hab. Alta densidad. Minorías significativas (30%).
  - Patrón Ramadán: Alteración drástica del ciclo. Incremento nocturno (01:00-04:00h) estimado en +35%.
  - Infraestructura: Estadio Rico Pérez. Eventos deportivos generan picos instantáneos al descanso y desalojo.

ZONA CENTRO (Rambla, Casco Antiguo, Mercado):
  - Economía: Dominancia del sector servicios y turismo de tránsito.
  - Flujo portuario: Capacidad de inyección de +1.500-5.000 pax flotantes por buque atracado.
  - Eventos: Epicentro de celebraciones mayores (Hogueras) con variaciones térmicas interanuales de +20-40%.

PLAYA DE SAN JUAN (Sectores costeros):
  - Economía: Alta estacionalidad. Alta densidad de instalaciones recreativas (piscinas).
  - Comportamiento: Sensibilidad extrema a variaciones térmicas prolongadas. Demanda pico x2.0 en periodo estival.
"""


# =============================================================================
# 12. MOTOR DE INFERENCIA LLM (Generación de Tensores)
# =============================================================================

def generar_factores_llm() -> tuple[dict, dict]:
    print("  Inicializando orquestación de datos y extracción de características en tiempo real...")

    clima      = obtener_clima_alicante()
    calendario = obtener_calendario()
    aire       = obtener_calidad_aire()
    fiestas    = obtener_fiestas_alicante()
    cruceros   = obtener_cruceros_alicante()
    vuelos     = obtener_vuelos_alicante()
    eventos    = obtener_eventos_ticketmaster()
    movilidad  = obtener_movilidad_mitma()
    obras      = obtener_obras_alicante()
    hotelero   = obtener_ocupacion_hotelera_ine()
    social     = escuchar_redes_sociales()

    contexto = {
        "clima": clima, "calendario": calendario, "aire": aire,
        "fiestas": fiestas, "cruceros": cruceros, "vuelos": vuelos,
        "eventos": eventos, "movilidad": movilidad, "obras": obras,
        "hotelero": hotelero, "social": social,
    }

    perfil  = calendario['perfil_dia']
    escolar = calendario['escolar']

    # Priors determinísticos (Base line para la Reductancia del LLM)
    priors = {
        "factor_cruceros":           cruceros['factor_impacto'],
        "factor_obras_construccion": obras['factor_obras'],
        "factor_ocupacion_hotelera": hotelero.get('factor_ocupacion', 1.0),
        "factor_vuelos_turismo":     vuelos['factor_estacional'],
        "factor_franja_manana":      perfil['f_manana'],
        "factor_franja_tarde":       perfil['f_tarde'],
        "factor_franja_noche":       perfil['f_noche'],
        "factor_fin_de_semana":      1.08 if calendario['es_fin_semana'] else 1.0,
        "factor_vacaciones_escolares": 1.10 if escolar['es_vacaciones'] else 1.0,
    }

    prompt = f"""
Operas como el motor de análisis heurístico para la red de abastecimiento hídrico de la ciudad de Alicante.
Objetivo Funcional: Extraer y clasificar variables del entorno para generar tensores multiplicativos.
Formato de Salida: ESTRUCTURA JSON ESTRICTA.

=== TELEMETRÍA DE ENTRADA (OSINT) ===

CLIMATOLOGÍA: {clima['resumen']}
  Cargas térmicas >28C: {clima['horas_sobre_28c']}h | Cargas críticas >32C: {clima['horas_sobre_32c']}h
  Sensación térmica pico (PM): {clima['sensacion_tarde']}C | Precipitación acumulada: {clima['lluvia_mm']}mm

CICLOS SOCIALES: {calendario['fecha']} ({calendario['dia_semana']})
  Estatus Festivo: {calendario['es_festivo']} ({calendario['nombre_festivo']})
  Ramadán en curso: {calendario['es_ramadan']} | Ciclo Lectivo: {escolar['periodo']} (Vacaciones activas: {escolar['es_vacaciones']})
  Perfil inercial ({perfil['nombre']}): AM x{perfil['f_manana']} / PM x{perfil['f_tarde']} / NOCHE x{perfil['f_noche']}

FESTIVIDADES: {fiestas['resumen']}
OPERATIVA PORTUARIA: {cruceros['resumen']} | Peso matemático: x{cruceros['factor_impacto']}
TRÁFICO AÉREO: {vuelos['resumen']} | Peso matemático: x{vuelos['factor_estacional']}
EVENTOS EN LA URBE: {eventos['resumen']}
ÍNDICE DE MOVILIDAD: {movilidad['resumen']}
OBRAS Y ALTERACIONES: {obras['resumen']} | Peso matemático: x{obras['factor_obras']}
DENSIDAD HOTELERA: {hotelero['resumen']}
PARTÍCULAS EN SUSPENSIÓN (PM): {aire['resumen']}
SEÑALES SOCIALES (OSINT): {social}

TOPOLOGÍA URBANA Y CONTEXTO ESPACIAL: {PERFIL_CIUDAD}

=== PRIORS MATEMÁTICOS ===
Los siguientes valores actúan como la predicción base algorítmica. 
Su modificación mediante el modelo generativo solo está autorizada bajo justificación explícita en los datos narrativos.
{json.dumps(priors, indent=2)}

=== REGLAS DE DOMINIO Y CLIPPING DE SEGURIDAD ===
- La combinación de factores se aplica de manera ponderada en el motor XGBoost, no de forma secuencial.
- Jornada ordinaria sin desviaciones: factor = 1.00.
- Límite operacional estricto: Desviaciones > 1.20 requieren validación cruzada con eventos físicos extremos.
- Rango absoluto de saturación del tensor: [0.70, 1.50].

=== REGLAS DE ENRUTAMIENTO ESPACIAL ===
- factor_cruceros          -> Enrutamiento restringido a ZONA_CENTRO.
- factor_ramadan_nocturno  -> Enrutamiento restringido a ZONA_NORTE, intervalo NOCHE.
- factor_calor_acumulado   -> Ponderación agravada en PLAYA_SAN_JUAN, intervalo TARDE.
- factor_zona_norte        -> Condicional a la detección de eventos en "Estadio Rico Pérez".
- factor_eventos           -> Localización geográfica según variable "zona" extraída del venue.
- factor_global            -> Impacto distribuido. Uso exclusivo para festividades de orden nacional o macro-climatología.

=== PROTOCOLO DE SALIDA ===
{{
    "factor_global":              1.00,
    "factor_zona_centro":         1.00,
    "factor_zona_norte":          1.00,
    "factor_playa_san_juan":      1.00,
    "factor_franja_manana":       {priors['factor_franja_manana']},
    "factor_franja_tarde":        {priors['factor_franja_tarde']},
    "factor_franja_noche":        {priors['factor_franja_noche']},
    "factor_fin_de_semana":       {priors['factor_fin_de_semana']},
    "factor_ramadan_nocturno":    1.00,
    "factor_calor_acumulado":     1.00,
    "factor_vacaciones_escolares":{priors['factor_vacaciones_escolares']},
    "factor_cruceros":            {priors['factor_cruceros']},
    "factor_vuelos_turismo":      {priors['factor_vuelos_turismo']},
    "factor_eventos":             1.00,
    "factor_movilidad_ciudad":    1.00,
    "factor_obras_construccion":  {priors['factor_obras_construccion']},
    "factor_ocupacion_hotelera":  {priors['factor_ocupacion_hotelera']},
    "alerta_averia_detectada":    false,
    "zona_averia":                null,
    "confianza":                  0.85,
    "razonamiento":               "Análisis ejecutivo: [Sintetizar vectores primarios y origen del desvío]"
}}
"""

    if not API_KEY:
        print("  [FALLBACK] API Key no detectada. Operando en modo determinista (priors).")
        return _factores_desde_priors(priors), contexto

    try:
        resp = cliente_ai.chat.completions.create(
            messages=[
                {"role": "system",
                 "content": (
                     "Capa de inferencia para Machine Learning. "
                     "Respuesta estricta en JSON sin bloques de código o formato markdown. "
                     "Rango operacional de tensores: [0.70, 1.50]. "
                     "Baseline: 1.0. Incrementos severos (>1.20) requieren fundamentación explícita."
                 )},
                {"role": "user", "content": prompt}
            ],
            model=MODELO,
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        factores = json.loads(resp.choices[0].message.content)

        CAMPOS_NUMERICOS = [
            "factor_global", "factor_zona_centro", "factor_zona_norte",
            "factor_playa_san_juan", "factor_franja_manana", "factor_franja_tarde",
            "factor_franja_noche", "factor_fin_de_semana", "factor_ramadan_nocturno",
            "factor_calor_acumulado", "factor_vacaciones_escolares", "factor_cruceros",
            "factor_vuelos_turismo", "factor_eventos", "factor_movilidad_ciudad",
            "factor_obras_construccion", "factor_ocupacion_hotelera",
        ]
        
        # Restricción estricta de tensores al rango [0.70, 1.50]
        for campo in CAMPOS_NUMERICOS:
            raw = factores.get(campo)
            if not isinstance(raw, (int, float)) or raw <= 0:
                fallback = priors.get(campo, 1.0)
                print(f"  [SAFETY_CLAMP] Parámetro {campo}={raw} inválido. Sobrescribiendo con prior: {fallback}")
                factores[campo] = fallback
            else:
                clamped = max(0.70, min(1.50, raw))
                if abs(clamped - raw) > 0.001:
                    print(f"  [SAFETY_CLAMP] Límite operacional excedido en {campo}: {raw:.3f} → {clamped:.3f}")
                factores[campo] = clamped

        conf = factores.get("confianza", 0.5)
        factores["confianza"] = max(0.0, min(1.0, float(conf) if isinstance(conf, (int, float)) else 0.5))

        return factores, contexto

    except json.JSONDecodeError as e:
        print(f"  [SYS_ERROR] Fallo de parseo JSON del modelo generativo: {e}. Retornando priors deterministas.")
        return _factores_desde_priors(priors), contexto
    except Exception as e:
        print(f"  [SYS_ERROR] Excepción en servicio LLM: {e}. Retornando priors deterministas.")
        return _factores_desde_priors(priors), contexto


def _factores_desde_priors(priors: dict) -> dict:
    base = {k: 1.0 for k in [
        "factor_global", "factor_zona_centro", "factor_zona_norte",
        "factor_playa_san_juan", "factor_ramadan_nocturno", "factor_calor_acumulado",
        "factor_eventos", "factor_movilidad_ciudad",
    ]}
    base.update(priors)
    base.update({
        "alerta_averia_detectada": False,
        "zona_averia": None,
        "confianza": 0.60,
        "razonamiento": "Inferencia bloqueada. Tensores procedentes exclusivamente de regresión matemática base."
    })
    return base


# =============================================================================
# 13. EJECUCIÓN DEL MÓDULO AISLADO (MODO DEBUG)
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SUBSISTEMA DE EXTRACCIÓN SOCIOLÓGICA (OSINT & LLM) — DIAGNÓSTICO")
    print("=" * 70)

    # Garantizar la existencia del directorio de salida
    os.makedirs(os.path.dirname(RUTA_OUTPUT_JSON), exist_ok=True)

    factores, ctx = generar_factores_llm()

    cal    = ctx['calendario']
    perfil = cal['perfil_dia']

    print("\n[+] CONTEXTO CAPTURADO (APIs Externas):")
    print(f"  Climatología      : {ctx['clima']['resumen']}")
    print(f"  Calidad Ambiental : {ctx['aire']['resumen']}")
    print(f"  Ciclo Temporal    : {cal['dia_semana']} | Festividad: {cal['es_festivo']} | Ramadán: {cal['es_ramadan']}")
    print(f"  Fase Escolar      : {cal['escolar']['periodo']}")
    print(f"  Celebraciones     : {ctx['fiestas']['resumen']}")
    print(f"  Tráfico Marítimo  : {ctx['cruceros']['resumen']}")
    print(f"  Tráfico Aéreo     : {ctx['vuelos']['resumen']}")
    print(f"  Agenda de Eventos : {ctx['eventos']['resumen']}")
    print(f"  Índice MITMA      : {ctx['movilidad']['resumen']}")
    print(f"  Obra Civil        : {ctx['obras']['resumen']}")
    print(f"  Densidad Turística: {ctx['hotelero']['resumen']}")

    print("\n" + "=" * 70)
    print("[+] VECTOR DE TENSORES EXTRAÍDO (A inyectar en modelo XGBoost):")
    print("=" * 70)
    print(json.dumps(factores, indent=4, ensure_ascii=False))

    output = {
        "timestamp": datetime.now().isoformat(),
        "factores":  factores,
        "contexto":  {k: ctx[k] for k in ["clima","calendario","fiestas","cruceros",
                                            "vuelos","eventos","movilidad","obras","hotelero"]}
    }
    
    with open(RUTA_OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)
    print(f"\n[OK] Snapshot de inferencia serializada en: {RUTA_OUTPUT_JSON}")