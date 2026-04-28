"""
Síntesis.py — Pipeline de Data Engineering e Imputación
==========================================================
Módulo responsable de la ingesta, limpieza y reconstrucción de series 
temporales hídricas. Cruza telemetría horaria parcial con facturación 
mensual para generar un dataset de 24 horas continuo (Ground Truth) 
mediante imputación basada en perfiles de consumo por tipología de sector.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURACIÓN DE RUTAS
# ==============================================================================
# Como este script está en /src, el BASE_DIR es el directorio padre (la raíz del proyecto)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

class DataImputer:
    
    def __init__(self):
        self.metricas = {}
        self.anomalias = []
        
    def cargar_datos_horarios(self, ruta: str) -> pd.DataFrame:
        """
        Ingesta y validación de series temporales de telemetría.
        Aplica correcciones de formato numérico europeo y filtra valores anómalos.
        """
        print("⏳ Iniciando ingesta de telemetría horaria...")
        parsed_data = []
        lineas_error = 0
        
        with open(ruta, 'r', encoding='utf-8') as f:
            for idx, linea in enumerate(f.readlines()[1:], start=2):
                linea = linea.strip()
                if not linea: continue
                
                try:
                    # Limpieza de caracteres de escape y formato numérico (comas a puntos)
                    if linea.startswith('"') and linea.endswith('"'):
                        partes = linea[1:-1].split(',', 2)
                        if len(partes) == 3:
                            caudal = float(partes[2].replace('"', '').replace(',', '.'))
                            parsed_data.append([partes[0], partes[1], caudal])
                    else:
                        partes = linea.split(',', 2)
                        if len(partes) == 3:
                            caudal = float(partes[2].replace(',', '.'))
                            parsed_data.append([partes[0], partes[1], caudal])
                except Exception as e:
                    lineas_error += 1
                    if lineas_error <= 5:
                        print(f"  ⚠️ Error de parseo en línea {idx}: {str(e)[:50]}")

        print(f"  ✅ {len(parsed_data)} registros procesados exitosamente. ({lineas_error} descartados)")
        
        df = pd.DataFrame(parsed_data, columns=['FECHA_HORA', 'SECTOR', 'CAUDAL_M3'])
        df['FECHA_HORA'] = pd.to_datetime(df['FECHA_HORA'], format='%d/%m/%Y %H:%M')
        df['Fecha'] = df['FECHA_HORA'].dt.date
        df['Mes']   = df['FECHA_HORA'].dt.month
        df['Hora']  = df['FECHA_HORA'].dt.hour
        df['Año']   = df['FECHA_HORA'].dt.year
        
        # VALIDACIÓN 1: Integridad física (Caudal >= 0)
        negativos = df[df['CAUDAL_M3'] < 0]
        if len(negativos) > 0:
            print(f"  ⚠️ Detectados {len(negativos)} registros inconsistentes (caudal negativo). Forzando a 0.")
            df.loc[df['CAUDAL_M3'] < 0, 'CAUDAL_M3'] = 0
        
        # VALIDACIÓN 2: Detección de Outliers Extremos Espaciales (Umbral: 3 IQR)
        for sector in df['SECTOR'].unique():
            datos_sector = df[df['SECTOR'] == sector]['CAUDAL_M3']
            Q1 = datos_sector.quantile(0.25)
            Q3 = datos_sector.quantile(0.75)
            IQR = Q3 - Q1
            limite_superior = Q3 + 3 * IQR
            
            outliers = df[(df['SECTOR'] == sector) & (df['CAUDAL_M3'] > limite_superior)]
            if len(outliers) > 0:
                self.anomalias.append({
                    'sector': sector,
                    'outliers': len(outliers),
                    'max_valor': outliers['CAUDAL_M3'].max(),
                    'limite': limite_superior
                })
        
        horas_disponibles = sorted(df['Hora'].unique())
        self.metricas['horas_origen'] = horas_disponibles
        
        return df

    def cargar_datos_mensuales(self, ruta: str, año_filtro: int = 2024) -> dict:
        """
        Ingesta de datos de facturación agrupados para calibración de volumen.
        Filtra por año para asegurar consistencia temporal con la telemetría.
        """
        print(f"⏳ Cargando matriz de facturación mensual (Calibración {año_filtro})...")
        df_m = pd.read_csv(ruta, sep=';', encoding='latin1') if 'aguas_corregido' in ruta else pd.read_csv(ruta)
        
        # Ajuste adaptativo del nombre de columna según la estructura del dataset
        col_fecha = 'Fecha (aaaa/mm/dd)' if 'Fecha (aaaa/mm/dd)' in df_m.columns else df_m.columns[0]
        col_consumo = 'Consumo (litros)' if 'Consumo (litros)' in df_m.columns else 'CAUDAL_M3'
        col_barrio = 'Barrio' if 'Barrio' in df_m.columns else 'BARRIO_AFECTADO'
        
        try:
            df_m[col_fecha] = pd.to_datetime(df_m[col_fecha], errors='coerce')
            df_m = df_m[df_m[col_fecha].dt.year == año_filtro]
            
            if len(df_m) == 0:
                print(f"  ⚠️ Advertencia: No se localizaron registros para {año_filtro}.")
                df_m = pd.read_csv(ruta)
                df_m[col_fecha] = pd.to_datetime(df_m[col_fecha], errors='coerce')
            else:
                print(f"  ✅ {len(df_m)} registros de facturación consolidados.")
            
            df_m['Mes'] = df_m[col_fecha].dt.month
            
            # Limpieza y conversión a metros cúbicos
            if df_m[col_consumo].dtype == 'O':
                df_m['Consumo_M3'] = pd.to_numeric(df_m[col_consumo].str.replace(',', ''), errors='coerce') / 1000.0
            else:
                df_m['Consumo_M3'] = df_m[col_consumo] / 1000.0
                
            df_agg = df_m.groupby([col_barrio, 'Mes'])['Consumo_M3'].sum().reset_index()
            self.metricas['consumo_total_mensual'] = df_agg['Consumo_M3'].sum()
            
            return dict(zip(zip(df_agg[col_barrio], df_agg.Mes), df_agg.Consumo_M3))
            
        except Exception as e:
            print(f"  ⚠️ Error procesando dataset mensual: {e}")
            return {}

    def sintetizar_24h(self, df_hora: pd.DataFrame, dict_mensual: dict, mapeo_sectores: dict) -> pd.DataFrame:
        """
        Imputación de horas faltantes cruzando el volumen mensual de facturación
        y distribuyéndolo estocásticamente mediante perfiles de consumo tipificados.
        """
        print("🧠 Ejecutando motor de síntesis para completitud de 24 horas...")
        
        # Perfiles base de distribución horaria de la demanda
        PERFILES = {
            'RESIDENCIAL': {
                13: 0.07, 14: 0.09, 15: 0.08, 16: 0.06, 17: 0.07, 18: 0.08,
                19: 0.10, 20: 0.13, 21: 0.14, 22: 0.10, 23: 0.06, 0: 0.02
            },
            'COMERCIAL': {
                13: 0.10, 14: 0.11, 15: 0.10, 16: 0.09, 17: 0.09, 18: 0.10,
                19: 0.11, 20: 0.09, 21: 0.06, 22: 0.05, 23: 0.03, 0: 0.01
            },
            'INDUSTRIAL': {
                13: 0.09, 14: 0.08, 15: 0.07, 16: 0.05, 17: 0.03, 18: 0.02,
                19: 0.01, 20: 0.01, 21: 0.01, 22: 0.01, 23: 0.01, 0: 0.00
            },
            'MIXTO': { 
                13: 0.08, 14: 0.10, 15: 0.09, 16: 0.07, 17: 0.06, 18: 0.07,
                19: 0.09, 20: 0.11, 21: 0.12, 22: 0.09, 23: 0.07, 0: 0.05
            }
        }
        
        TIPO_SECTOR = {
            "CENTRO COMERCIAL GRAN VÍA": 'COMERCIAL',
            "CIUDAD DEPORTIVA DL": 'MIXTO',
            "ALIPARK DL": 'COMERCIAL',
            # Default: MIXTO para sectores sin categorización explícita
        }
        
        df_hora['Barrio_Asignado'] = df_hora['SECTOR'].map(mapeo_sectores)
        
        vol_sector_mes = df_hora.groupby(['SECTOR', 'Mes'])['CAUDAL_M3'].sum().to_dict()
        vol_barrio_mes = df_hora.groupby(['Barrio_Asignado', 'Mes'])['CAUDAL_M3'].sum().to_dict()
        vol_sector_dia = df_hora.groupby(['SECTOR', 'Mes', 'Fecha'])['CAUDAL_M3'].sum().reset_index()

        datos_sinteticos = []
        sectores_con_mapeo = 0
        sectores_sin_mapeo = 0

        for _, fila in vol_sector_dia.iterrows():
            sector = fila['SECTOR']
            mes = fila['Mes']
            fecha = fila['Fecha']
            vol_dia_mañana = fila['CAUDAL_M3']
            
            barrio = mapeo_sectores.get(sector)
            vol_mes_mañana_sector = vol_sector_mes.get((sector, mes), 1e-6)
            
            vol_faltante_tarde_mes = 0
            usar_imputacion_fallback = False
            metodo_usado = "indeterminado"
            
            # Estrategia de emparejamiento y cálculo de pesos
            if barrio is not None:
                total_facturado_barrio = dict_mensual.get((barrio, mes), 0)
                vol_mes_mañana_barrio = vol_barrio_mes.get((barrio, mes), 1e-6)
                
                peso_del_sector = vol_mes_mañana_sector / vol_mes_mañana_barrio if vol_mes_mañana_barrio > 0 else 0
                total_estimado_sector = total_facturado_barrio * peso_del_sector
                vol_faltante_tarde_mes = total_estimado_sector - vol_mes_mañana_sector
                
                if vol_faltante_tarde_mes <= 0:
                    usar_imputacion_fallback = True
                    metodo_usado = "imputacion_heuristica"
                else:
                    sectores_con_mapeo += 1
                    metodo_usado = "calibracion_facturacion"
            else:
                usar_imputacion_fallback = True
                metodo_usado = "imputacion_fallback"
                
            if usar_imputacion_fallback:
                sectores_sin_mapeo += 1
                # Estimación estándar basada en la asunción de distribución 45% (AM) / 55% (PM)
                vol_faltante_tarde_mes = (vol_mes_mañana_sector / 0.45) * 0.55

            vol_faltante_tarde_mes = max(0, vol_faltante_tarde_mes)
            
            ratio_dia = vol_dia_mañana / vol_mes_mañana_sector if vol_mes_mañana_sector > 0 else 0
            vol_faltante_hoy = vol_faltante_tarde_mes * ratio_dia
            
            tipo = TIPO_SECTOR.get(sector, 'MIXTO')
            perfil = PERFILES[tipo]
            
            for hora, peso in perfil.items():
                datos_sinteticos.append({
                    'FECHA_HORA': pd.to_datetime(f"{fecha} {hora}:00:00"),
                    'SECTOR': sector,
                    'CAUDAL_M3': round(vol_faltante_hoy * peso, 3),
                    'METODO': metodo_usado
                })

        print(f"  ✅ Sectores calibrados con facturación: {sectores_con_mapeo}")
        print(f"  ⚠️ Sectores procesados mediante fallback heurístico: {sectores_sin_mapeo}")
        
        df_sintetico = pd.DataFrame(datos_sinteticos)
        df_completo = pd.concat([
            df_hora[['FECHA_HORA', 'SECTOR', 'CAUDAL_M3']].assign(METODO='telemetria_real'),
            df_sintetico
        ])
        df_completo = df_completo.sort_values(['SECTOR', 'FECHA_HORA']).reset_index(drop=True)
        
        # Registro de métricas de validación
        self.metricas['total_original'] = df_hora['CAUDAL_M3'].sum()
        self.metricas['total_sintetico'] = df_sintetico['CAUDAL_M3'].sum() if not df_sintetico.empty else 0
        self.metricas['ratio_tarde_mañana'] = self.metricas['total_sintetico'] / self.metricas['total_original'] if self.metricas['total_original'] > 0 else 0
        
        return df_completo

    def generar_reporte_calidad(self):
        """Genera el reporte final de validación de la síntesis del dataset."""
        print("\n" + "="*80)
        print("📊 REPORTE DE CALIDAD DE DATOS (DATA QUALITY AUDIT)")
        print("="*80)
        
        print(f"\n🔢 MÉTRICAS DE VOLUMEN GLOBAL:")
        print(f"  • Telemetría verificada (AM):   {self.metricas.get('total_original', 0):,.0f} m³")
        print(f"  • Imputación calculada (PM):    {self.metricas.get('total_sintetico', 0):,.0f} m³")
        print(f"  • Proporción PM/AM:             {self.metricas.get('ratio_tarde_mañana', 0):.2%}")
        
        if self.metricas.get('ratio_tarde_mañana', 0) < 0.5 or self.metricas.get('ratio_tarde_mañana', 0) > 2.0:
            print(f"  ⚠️ ALERTA: La proporción de imputación se encuentra fuera del intervalo esperado (0.5 - 2.0).")
        
        if self.anomalias:
            print(f"\n⚠️ ANOMALÍAS ESPACIALES (OUTLIERS DETECTADOS): {len(self.anomalias)} sectores")
            for a in self.anomalias[:5]:
                print(f"  • {a['sector']}: {a['outliers']} registros > {a['limite']:.1f} m³ (Máximo histórico: {a['max_valor']:.1f})")
        
        print("\n✅ Proceso de consolidación finalizado exitosamente.")
        print("="*80)


# ==============================================================================
# CONFIGURACIÓN GEOGRÁFICA Y EJECUCIÓN
# ==============================================================================
MAPEO_SECTORES = {
    "1 CIUDAD JARDÍN": "31-CIUDAD JARDIN",
    "ALIPARK DL": "8-ALIPARK",
    "ALTOZANO": "19-ALTOZANO",
    "BAHÍA LOS PINOS": None, 
    "BENALÚA DL": "1-BENALUA",
    "Bº GRANADA 1": None,
    "Bº LOS ÁNGELES": "6-LOS ANGELES",
    "CABO HUERTAS - PLAYA": "40-CABO DE LAS HUERTAS",
    "CENTRO COMERCIAL GRAN VÍA": None,
    "CIUDAD DEPORTIVA DL": "11-CIUDAD DE ASIS",
    "COLONIA REQUENA": "34-COLONIA REQUENA",
    "COLONIA ROMANA": "34-COLONIA REQUENA",
    "CONDOMINA": "34-COLONIA REQUENA",
    "Campoamor Alto": "5-CAMPOAMOR",
    "DIPUTACIÓN DL": "14-ENSANCHE DIPUTACION",
    "Depósito Los Ángeles": "6-LOS ANGELES",
    "GARBINET NORTE 1": "19-GARBINET",
    "INFORMACIÓN DL": None,
    "LONJA": None,
    "LONJA DL": None,
    "Les Palmeretes": "28-EL PALMERAL",
    "MATADERO": "4-MERCADO",
    "MERCADO DL": "4-MERCADO",
    "MUCHAVISTA - P.A.U. 5": None,
    "MUELLE GRANELES DL": "6-LOS ANGELES",
    "MUELLE LEVANTE DL": None,
    "O.A.M.I 1": None,
    "P.A.U. 1 (norte+sur)": None,
    "P.A.U. 2": None,
    "PARQUE LO MORANT": None,
    "PLAYA DE SAN JUAN 1": "41-PLAYA DE SAN JUAN",
    "PZA. MONTAÑETA": "FONTCALENT",
    "Pla-Hospital": None,
    "Postiguet": "55-PUERTO",
    "RABASA DL": "20-RABASA",
    "SANTO DOMINGO DL": "24-SAN BLAS - SANTO DOMINGO",
    "SH_Demo": "38-VISTAHERMOSA",
    "TOBO": "21-TOMBOLA",
    "VALLONGA GLOBAL": "PDA VALLONGA",
    "VALLONGA-TOLON DL": "PDA VALLONGA",
    "VILLAFRANQUEZA": "VILLAFRANQUEZA",
    "VIRGEN DEL CARMEN 1000 Viv": "35-VIRGEN DEL CARMEN",
    "VIRGEN DEL REMEDIO": "32-VIRGEN DEL REMEDIO"
}

if __name__ == "__main__":
    
    # Verificación de estructura de directorios
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    imputer = DataImputer()
    
    # Ingesta
    ruta_horarios = os.path.join(RAW_DIR, 'caudales_horarios.csv')
    df_h = imputer.cargar_datos_horarios(ruta_horarios)
    
    ruta_mensuales = os.path.join(RAW_DIR, 'aguas_corregido_v2_Sheet1_.csv')
    dict_m = imputer.cargar_datos_mensuales(ruta_mensuales, año_filtro=2024)
    
    # Procesamiento
    df_final = imputer.sintetizar_24h(df_h, dict_m, MAPEO_SECTORES)
    imputer.generar_reporte_calidad()
    
    # Exportación
    ruta_exportacion = os.path.join(PROCESSED_DIR, 'Dataset_24H_Mejorado_V2.csv')
    df_final.to_csv(ruta_exportacion, index=False)
    print(f"\n💾 Dataset consolidado exportado en: {ruta_exportacion}")