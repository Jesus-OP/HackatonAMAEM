"""
agente_llm.py — VERSIÓN CORREGIDA
===================================
Cambios respecto a la versión original:
  1. _eventos_simulados() eliminado — ya NO inventa un partido del Hércules.
     Ahora el fallback devuelve "Sin eventos confirmados hoy."
  2. cargar_eventos_csv() — nueva función que lee el CSV real de AMAEM
     (aguas_corregido_v2_Sheet1_.csv) y filtra los eventos del día actual.
     El agente usa primero Ticketmaster, si falla usa el CSV, si falla devuelve vacío.
  3. El prompt del LLM tiene rangos calibrados más estrictos:
     - Día normal sin nada especial → todos los factores = 1.0
     - Máximo factor individual en caso extremo → 1.50 (antes 2.50)
     Esto evita que el LLM infle factores y genere falsos estrés.
  4. API keys movidas a variables de entorno (sin fallback hardcodeado).
"""

import requests
import xml.etree.ElementTree as ET
from groq import Groq
from datetime import datetime, date
from bs4 import BeautifulSoup
import json
import os
import pandas as pd

def cargar_config(key_name):
    # 1. Intento por Streamlit (Nube)
    try:
        import streamlit as st
        if key_name in st.secrets:
            return st.secrets[key_name]
    except:
        pass
    
    # 2. Intento por Variables de Entorno (Local/WinPython)
    return os.getenv(key_name, "")

# Ahora las cargas así de limpio:
API_KEY        = cargar_config("GROQ_API_KEY")
TICKETMASTER_KEY = cargar_config("TICKETMASTER_KEY")
MODELO           = "llama-3.3-70b-versatile"

# Ruta al CSV de eventos reales de AMAEM (en la misma carpeta que este script)
RUTA_CSV_EVENTOS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "aguas_corregido_v2_Sheet1_.csv")

if API_KEY:
    cliente_ai = Groq(api_key=API_KEY)
else:
    print("⚠️  GROQ_API_KEY no configurada. Ejecutando en modo fallback (priors Python).")


# =============================================================================
# 1. CLIMA
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
        print(f"  [WARN] Clima: {e}")
        return {"temp_max":20,"temp_min":14,"lluvia_mm":0,"hora_pico_calor":15,
                "horas_sobre_28c":0,"horas_sobre_32c":0,"sensacion_tarde":20,
                "resumen":"Primavera tipica 20C (estimado)"}

def _resumir_clima(temp, lluvia):
    if temp > 35:   return f"Ola de calor: {temp}C"
    if temp > 30:   return f"Calor alto: {temp}C"
    if lluvia > 10: return f"Lluvia intensa: {lluvia}mm, {temp}C"
    if lluvia > 0:  return f"Lluvia leve: {lluvia}mm, {temp}C"
    return f"Despejado: {temp}C"

def _sensacion_termica(temp, humedad, viento):
    if temp < 27: return temp - viento * 0.1
    return (-8.784695 + 1.61139411*temp + 2.338549*humedad
            - 0.14611605*temp*humedad - 0.01230809*temp**2
            - 0.01642482*humedad**2 + 0.00221173*temp**2*humedad
            + 0.00072546*temp*humedad**2 - 0.00000358*temp**2*humedad**2)


# =============================================================================
# 2. CALENDARIO
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
        print(f"  [WARN] Festivos: {e}")
    return res

def _es_ramadan(fecha):
    rangos = {2025:((3,1),(3,30)), 2026:((2,18),(3,19)), 2027:((2,8),(3,9))}
    r = rangos.get(fecha.year)
    if not r: return False
    return datetime(fecha.year,r[0][0],r[0][1]) <= fecha <= datetime(fecha.year,r[1][0],r[1][1])

def _estado_escolar(fecha):
    m, d = fecha.month, fecha.day
    vac, per = False, "Curso escolar"
    if (m==6 and d>=15) or m in [7,8] or (m==9 and d<=10): vac, per = True, "Verano escolar"
    elif (m==12 and d>=23) or (m==1 and d<=7):              vac, per = True, "Vacaciones Navidad"
    elif _es_semana_santa(fecha):                            vac, per = True, "Semana Santa"
    patron = ("Pico matutino retrasado 9h-10h." if vac else
              "Pico matutino 7h-8h30." if fecha.weekday()<5 else "Fin de semana.")
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
        0:{"nombre":"Lunes",    "f_man":1.10,"f_tar":0.95,"f_noc":0.90,"nota":"Lavadoras post-finde."},
        1:{"nombre":"Martes",   "f_man":1.00,"f_tar":1.00,"f_noc":0.95,"nota":"Dia estandar."},
        2:{"nombre":"Miercoles","f_man":1.00,"f_tar":1.00,"f_noc":0.95,"nota":"Patron normal."},
        3:{"nombre":"Jueves",   "f_man":1.00,"f_tar":1.02,"f_noc":1.02,"nota":"Pre-finde."},
        4:{"nombre":"Viernes",  "f_man":1.00,"f_tar":1.05,"f_noc":1.10,"nota":"Tarde-noche ocio."},
        5:{"nombre":"Sabado",   "f_man":0.90,"f_tar":1.10,"f_noc":1.15,"nota":"Limpieza, piscinas."},
        6:{"nombre":"Domingo",  "f_man":0.85,"f_tar":1.15,"f_noc":0.90,"nota":"Paellas 12h-15h."},
    }[fecha.weekday()]
    return {"nombre":p["nombre"],"f_manana":p["f_man"],"f_tarde":p["f_tar"],
            "f_noche":p["f_noc"],"nota":p["nota"]}


# =============================================================================
# 3. FIESTAS LOCALES ALICANTE
# =============================================================================

FIESTAS_ALICANTE = [
    (6,20,24,"Hogueras de San Juan",1.30,"Masiva afluencia. Pico agua noche."),
    (6,23,23,"Noche de la Crema",   1.40,"+100.000 personas. Noche mas intensa."),
    (6,17,19,"Pre-Hogueras",        1.10,"Ambiente festivo. Barracas abiertas."),
    (12,25,26,"Navidad",            0.80,"Residencial alto, comercios cerrados."),
    (12,31,31,"Nochevieja",         1.20,"Concentracion masiva zona centro."),
    (1,1,1,"Ano Nuevo",             0.75,"Dia baja actividad."),
    (3,19,19,"San Jose",            0.95,"Festivo autonomico."),
    (8,1,8,"Moros y Cristianos",    1.10,"Afluencia regional zona norte."),
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
            f"FIESTA HOY: {fh['nombre']} (x{fh['factor']}) — {fh['detalle']}" if fh else
            f"FIESTA MANANA: {fm['nombre']}" if fm else
            "SEMANA HOGUERAS: Ambiente festivo general" if hogueras else
            "Sin fiestas locales destacadas."
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
                "resumen":f"CALIMA: {polvo} ug/m3" if polvo>50 else "Aire limpio"}
    except Exception as e:
        print(f"  [WARN] Aire: {e}")
        return {"polvo_ug_m3":0,"alerta_calima":False,"resumen":"Normal (estimado)"}


# =============================================================================
# 5. CRUCEROS — Puerto de Alicante
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
        # FIX: factor más conservador (antes llegaba a 1.35 con apenas 3500 pax)
        factor = round(min(1.0 + (total / 15000) * 0.25, 1.25), 2)
        return {
            "cruceros_hoy": cruceros_hoy, "num_cruceros": len(cruceros_hoy),
            "pasajeros_totales": total, "factor_impacto": factor,
            "resumen": (
                f"{len(cruceros_hoy)} crucero(s) — ~{total:,} pax extra en centro."
                if cruceros_hoy else "Sin cruceros en puerto hoy."
            )
        }
    except Exception as e:
        print(f"  [WARN] Cruceros: {e}")
        # FIX: fallback neutro, sin inventar cruceros
        return {"cruceros_hoy":[],"num_cruceros":0,"pasajeros_totales":0,
                "factor_impacto":1.0,"resumen":"Sin cruceros confirmados hoy."}

def _est_pax(nombre):
    n = nombre.lower()
    if any(x in n for x in ["symphony","wonder","icon","oasis","allure"]): return 5000
    if any(x in n for x in ["msc","costa","carnival","celebrity"]):         return 3000
    return 1500


# =============================================================================
# 6. VUELOS
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
        # FIX: factor estacional más conservador
        factor_est = 1.20 if mes in [7,8] else 1.10 if mes in [6,9] else 1.0
        return {
            "vuelos_aterrizando": len(aterrizando),
            "vuelos_aproximacion": len(aproximacion),
            "pasajeros_est_hora": len(aproximacion)*150,
            "factor_estacional": factor_est,
            "resumen": (
                f"{len(aterrizando)} aterrizando | {len(aproximacion)} en aprox. "
                f"(~{len(aproximacion)*150} pax/h) | Factor estacional x{factor_est}"
            )
        }
    except Exception as e:
        print(f"  [WARN] OpenSky: {e}")
        mes = datetime.now().month
        factor = 1.20 if mes in [7,8] else 1.10 if mes in [6,9] else 1.0
        return {"vuelos_aterrizando":0,"vuelos_aproximacion":0,"pasajeros_est_hora":0,
                "factor_estacional":factor,"resumen":f"Factor estacional x{factor} (mes {mes})"}


# =============================================================================
# 7. EVENTOS — CSV real de AMAEM + Ticketmaster como enriquecimiento
# =============================================================================

def cargar_eventos_csv_hoy() -> list[dict]:
    """
    FIX PRINCIPAL: Lee el CSV real de eventos de AMAEM y filtra los de hoy.
    Este CSV tiene datos reales de cruceros, Hogueras, Ramadán, partidos mundiales, etc.
    Sustituye al _eventos_simulados() que inventaba un partido del Hércules.
    """
    eventos_hoy = []
    if not os.path.exists(RUTA_CSV_EVENTOS):
        print(f"  [WARN] CSV eventos no encontrado en: {RUTA_CSV_EVENTOS}")
        return []

    try:
        df = pd.read_csv(RUTA_CSV_EVENTOS, sep=';', encoding='latin1')
        df['FECHA_INICIO'] = pd.to_datetime(df['FECHA_INICIO'], format='mixed', dayfirst=False)
        df['FECHA_FIN']    = pd.to_datetime(df['FECHA_FIN'],    format='mixed', dayfirst=False)

        hoy = datetime.now().date()

        # Un evento aplica hoy si hoy está en el rango [FECHA_INICIO, FECHA_FIN]
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
                "fuente":  "CSV_AMAEM",
            })

        if eventos_hoy:
            print(f"  [CSV] {len(eventos_hoy)} evento(s) encontrado(s) para hoy en el CSV de AMAEM.")
        else:
            print("  [CSV] Sin eventos en el CSV para la fecha de hoy.")

    except Exception as e:
        print(f"  [WARN] Error leyendo CSV eventos: {e}")

    return eventos_hoy

def _aforo_desde_impacto(impacto):
    """Convierte la escala de impacto del CSV (1-5) a aforo estimado."""
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
    """
    Intenta Ticketmaster. Si falla, usa el CSV de AMAEM como fuente de verdad.
    FIX: eliminado _eventos_simulados() que inventaba partido del Hércules.
    """
    # 1. Intentar Ticketmaster si hay key
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
            # Enriquecer con CSV de AMAEM
            eventos_csv = cargar_eventos_csv_hoy()
            eventos_csv_no_dup = [e for e in eventos_csv
                                  if not any(e['nombre'].lower() in ev['nombre'].lower()
                                             for ev in eventos)]
            eventos.extend(eventos_csv_no_dup)
            factor = round(min(1.0 + sum(e['aforo'] for e in eventos)/20000*0.10, 1.30), 2)
            return {
                "eventos_hoy": eventos, "num_eventos": len(eventos), "factor_evento": factor,
                "resumen": (
                    " | ".join([f"{e['nombre']} ({e['hora']}, ~{e['aforo']} asist., zona {e['zona']})"
                                for e in eventos])
                    if eventos else "Sin eventos confirmados hoy en Alicante."
                )
            }
        except Exception as e:
            print(f"  [WARN] Ticketmaster: {e} — usando CSV AMAEM")

    # 2. Fallback: solo CSV de AMAEM
    eventos_csv = cargar_eventos_csv_hoy()
    factor = round(min(1.0 + sum(e['aforo'] for e in eventos_csv)/20000*0.10, 1.25), 2)
    return {
        "eventos_hoy": eventos_csv, "num_eventos": len(eventos_csv), "factor_evento": factor,
        "resumen": (
            " | ".join([f"{e['nombre']} ({e['hora']}, zona {e['zona']}, impacto {e['impacto']}/5)"
                        for e in eventos_csv])
            if eventos_csv else "Sin eventos confirmados hoy en Alicante."
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
# 8. MOVILIDAD MITMA
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
            "resumen": f"Indice movilidad Alicante: {indice} ({variacion:+.1f}%). Factor x{factor}"
        }
    except Exception as e:
        print(f"  [WARN] MITMA API: {e} — usando estimacion estacional")
        return _movilidad_estacional()

def _movilidad_estacional() -> dict:
    mes = datetime.now().month
    # FIX: índices más conservadores (antes abril=95 llegaba a x0.95, ahora es neutro)
    indice_por_mes = {
        1:88, 2:90, 3:95, 4:100, 5:102,
        6:108, 7:130, 8:140, 9:115, 10:100,
        11:92, 12:95
    }
    indice = indice_por_mes.get(mes, 100)
    factor = round(indice / 100, 2)
    return {
        "indice_movilidad": indice, "variacion_mensual": 0, "factor_movilidad": factor,
        "resumen": f"Movilidad estimada {indice}/100 (mes {mes}). Factor x{factor}."
    }


# =============================================================================
# 9. OBRAS
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
                    "descripcion": obra.get("descripcion", "Obra mayor"),
                    "direccion":   obra.get("direccion", ""),
                    "zona":        zona, "estado": obra.get("estado",""),
                })
        factor = round(min(1.0 + len(obras_activas)*0.01, 1.10), 2)
        return {
            "obras_activas": obras_activas[:5], "total_obras": len(obras_activas),
            "factor_obras": factor,
            "resumen": (
                f"{len(obras_activas)} obras mayores activas. Factor x{factor}."
                if obras_activas else "Sin datos de obras activas."
            )
        }
    except Exception as e:
        print(f"  [WARN] Obras: {e}")
        return {"obras_activas":[],"total_obras":0,"factor_obras":1.0,
                "resumen":"Datos de obras no disponibles."}

def _detectar_zona_obra(texto: str) -> str:
    t = texto.lower()
    if any(x in t for x in ["playa","san juan","cabo","postiguet"]): return "playa"
    if any(x in t for x in ["norte","carolinas","remedio","florida"]): return "norte"
    return "centro"


# =============================================================================
# 10. INE — Ocupación hotelera
# =============================================================================

def obtener_ocupacion_hotelera_ine() -> dict:
    try:
        url = "https://servicios.ine.es/wstempus/js/ES/DATOS_SERIE/IH9009?nult=13"
        r   = requests.get(url, timeout=8).json()
        datos_serie = r.get("Data", [])
        if not datos_serie: raise ValueError("Serie INE vacia")
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
                f"INE EOH: {int(viajeros_ultimo):,} viajeros "
                f"({variacion_mensual:+.1f}% vs mes ant). Factor x{factor}"
            )
        }
    except Exception as e:
        print(f"  [WARN] INE EOH: {e} — estimacion estacional")
        return _ocupacion_estacional()

def _ocupacion_estacional() -> dict:
    mes = datetime.now().month
    # FIX: factores más conservadores
    ocupacion_mes = {
        1:60, 2:63, 3:70, 4:80, 5:85,
        6:92, 7:115, 8:125, 9:105, 10:85,
        11:65, 12:70
    }
    ocup   = ocupacion_mes.get(mes, 80)
    factor = round(ocup / 100, 2)
    return {
        "viajeros_ultimo_mes": 0, "variacion_mensual_pct": 0, "variacion_anual_pct": 0,
        "factor_ocupacion": factor, "periodo": f"Estimacion mes {mes}",
        "resumen": f"Ocupacion hotelera estimada: {ocup}/100 (mes {mes}). Factor x{factor}."
    }


# =============================================================================
# 11. REDES SOCIALES Y NOTICIAS
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
                fragmentos.append(f"RSS NOTICIAS: {' | '.join(titulares)}")
                break
        except Exception as e:
            print(f"  [WARN] RSS {nombre}: {e}")

    if not any('RSS' in f for f in fragmentos):
        fragmentos.append("RSS NOTICIAS: No disponible.")

    try:
        r = requests.get("https://www.reddit.com/r/alicante/new.json?limit=10",
                         headers={'User-Agent':'HackathonAguasAlicante/1.0'}, timeout=5).json()
        kw = ['agua','calor','corte','averia','inundacion','sequia','lluvia','heat','water']
        rel = [p['data']['title'] for p in r['data']['children']
               if any(w in p['data']['title'].lower() for w in kw)]
        fragmentos.append(f"REDDIT: {' | '.join(rel[:3])}" if rel else "REDDIT: Sin posts relevantes.")
    except Exception as e:
        print(f"  [WARN] Reddit: {e}")
        fragmentos.append("REDDIT: Sin anomalias.")

    return "\n".join(fragmentos)


# =============================================================================
# PERFIL CIUDAD
# =============================================================================

PERFIL_CIUDAD = """
ZONA NORTE (Virgen del Remedio, Carolinas):
  - 85.000 hab. 30% musulmana. Ramadan: pico nocturno 1h-4h (+35% consumo).
  - Estadio Rico Perez: partidos = pico brusco al descanso y post-partido.

ZONA CENTRO (Rambla, Casco Antiguo, Mercado):
  - Turismo intenso. Cruceros atracan en muelle adyacente (+1.500-5.000 pax).
  - Hogueras, conciertos, ferias = +20-40%.

PLAYA SAN JUAN:
  - Residencial alto + turismo veraniego. Muchas piscinas privadas.
  - Julio-agosto: demanda x2.0 vs media anual. Aeropuerto a 10km.
"""


# =============================================================================
# 12. LLM -> FACTORES PARA ML
# =============================================================================

def generar_factores_llm() -> tuple[dict, dict]:
    print("  Recopilando datos en tiempo real...")

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

    # Priors calculados desde Python (el LLM solo ajusta si hay justificación real)
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
Eres un motor de prediccion de demanda hidrica para Aguas de Alicante.
Tu UNICA tarea: devolver multiplicadores numericos como JSON puro.

=== DATOS CAPTURADOS ===

CLIMA: {clima['resumen']}
  Horas >28C: {clima['horas_sobre_28c']}h | Horas >32C: {clima['horas_sobre_32c']}h
  Sensacion termica tarde: {clima['sensacion_tarde']}C | Lluvia: {clima['lluvia_mm']}mm

CALENDARIO: {calendario['fecha']} ({calendario['dia_semana']})
  Festivo: {calendario['es_festivo']} ({calendario['nombre_festivo']})
  Ramadan: {calendario['es_ramadan']} | Escolar: {escolar['periodo']} (vacaciones: {escolar['es_vacaciones']})
  Perfil {perfil['nombre']}: manana x{perfil['f_manana']} / tarde x{perfil['f_tarde']} / noche x{perfil['f_noche']}

FIESTAS LOCALES: {fiestas['resumen']}
CRUCEROS: {cruceros['resumen']} | Prior calculado: x{cruceros['factor_impacto']}
VUELOS: {vuelos['resumen']} | Prior estacional: x{vuelos['factor_estacional']}
EVENTOS HOY (fuentes verificadas): {eventos['resumen']}
MOVILIDAD: {movilidad['resumen']}
OBRAS ACTIVAS: {obras['resumen']} | Prior calculado: x{obras['factor_obras']}
OCUPACION HOTELERA: {hotelero['resumen']}
CALIDAD AIRE: {aire['resumen']}
OSINT SOCIAL: {social}

PERFIL CIUDAD: {PERFIL_CIUDAD}

=== PRIORS CALCULADOS POR PYTHON ===
Estos valores estan calculados matematicamente. Usaos como base y ajustalos
SOLO si los datos narrativos justifican claramente una desviacion:
{json.dumps(priors, indent=2)}

=== REGLAS CRITICAS DE CALIBRACION ===
⚠️  IMPORTANTE: Los factores NO se multiplican entre si — se combinan como media ponderada.
Por tanto, un factor de 1.20 ya es un ajuste importante (representa +20% en esa variable).
- Dia laborable normal sin eventos = TODOS los factores = 1.0
- Solo superar 1.20 si hay evidencia SOLIDA (evento masivo confirmado, ola de calor real)
- Solo superar 1.35 para eventos EXCEPCIONALES (Hogueras, ola extrema >37C, megacrucero)
- Maximo absoluto por factor individual: 1.50 (antes era 2.50 — era un error de calibracion)

=== REGLAS DE APLICACION ===
- factor_cruceros          -> SOLO ZONA_CENTRO
- factor_ramadan_nocturno  -> SOLO ZONA_NORTE + franja NOCHE
- factor_calor_acumulado   -> PLAYA_SAN_JUAN y franja TARDE
- factor_zona_norte        -> Solo si hay evento confirmado en Estadio Rico Perez HOY
- factor_eventos           -> Segun zona del venue real (no inventado)
- factor_global            -> Solo para festivos nacionales o fenomenos que afectan toda la ciudad

=== FORMATO SALIDA (SOLO JSON VALIDO, SIN MARKDOWN) ===
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
    "razonamiento":               "Factores dominantes: [lista los 2-3 mas importantes y por que]"
}}
"""

    if not API_KEY:
        print("  [FALLBACK] Sin API key — usando priors Python.")
        return _factores_desde_priors(priors), contexto

    try:
        resp = cliente_ai.chat.completions.create(
            messages=[
                {"role": "system",
                 "content": (
                     "Motor matematico hidrico. "
                     "Devuelve SOLO JSON valido sin markdown. "
                     "Factores entre 0.70 y 1.50. "
                     "1.0 = consumo normal. Solo supera 1.20 con evidencia solida."
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
        # FIX: clamping más estricto [0.70, 1.50] en lugar de [0.30, 2.50]
        for campo in CAMPOS_NUMERICOS:
            raw = factores.get(campo)
            if not isinstance(raw, (int, float)) or raw <= 0:
                fallback = priors.get(campo, 1.0)
                print(f"  [CLAMP] {campo}={raw} invalido → prior {fallback}")
                factores[campo] = fallback
            else:
                clamped = max(0.70, min(1.50, raw))
                if abs(clamped - raw) > 0.001:
                    print(f"  [CLAMP] {campo}={raw:.3f} → {clamped:.3f}")
                factores[campo] = clamped

        conf = factores.get("confianza", 0.5)
        factores["confianza"] = max(0.0, min(1.0, float(conf) if isinstance(conf, (int, float)) else 0.5))

        return factores, contexto

    except json.JSONDecodeError as e:
        print(f"  [ERROR] JSON invalido del LLM: {e} — usando priors Python")
        return _factores_desde_priors(priors), contexto
    except Exception as e:
        print(f"  [ERROR] LLM: {e} — usando priors Python")
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
        "razonamiento": "Modo fallback — factores calculados desde fuentes Python sin LLM."
    })
    return base


# =============================================================================
# 13. EJECUCION
# =============================================================================

if __name__ == "__main__":
    print("=" * 55)
    print("SISTEMA DE PREDICCION HIDRICA — ALICANTE")
    print("=" * 55)

    factores, ctx = generar_factores_llm()

    cal    = ctx['calendario']
    perfil = cal['perfil_dia']

    print("\nCONTEXTO CAPTURADO:")
    print(f"  Clima      : {ctx['clima']['resumen']}")
    print(f"  Aire       : {ctx['aire']['resumen']}")
    print(f"  Dia        : {cal['dia_semana']} | Festivo: {cal['es_festivo']} | Ramadan: {cal['es_ramadan']}")
    print(f"  Escolar    : {cal['escolar']['periodo']}")
    print(f"  Fiestas    : {ctx['fiestas']['resumen']}")
    print(f"  Cruceros   : {ctx['cruceros']['resumen']}")
    print(f"  Eventos    : {ctx['eventos']['resumen']}")
    print(f"  Movilidad  : {ctx['movilidad']['resumen']}")
    print(f"  Obras      : {ctx['obras']['resumen']}")
    print(f"  Hotelero   : {ctx['hotelero']['resumen']}")

    print("\n" + "=" * 55)
    print("MATRIZ DE FACTORES (-> MODELO ML):")
    print("=" * 55)
    print(json.dumps(factores, indent=4, ensure_ascii=False))

    output = {
        "timestamp": datetime.now().isoformat(),
        "factores":  factores,
        "contexto":  {k: ctx[k] for k in ["clima","calendario","fiestas","cruceros",
                                            "vuelos","eventos","movilidad","obras","hotelero"]}
    }
    with open("factores_hoy.json","w",encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)
    print(f"\nGuardado en: factores_hoy.json")