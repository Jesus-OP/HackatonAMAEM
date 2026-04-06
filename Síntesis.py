import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# VERSIÓN MEJORADA: Con validaciones, logging y manejo robusto de errores
# ==============================================================================

class AnalizadorCaudalMejorado:
    
    def __init__(self):
        self.metricas = {}
        self.anomalias = []
        
    def cargar_datos_horarios(self, ruta):
        """Parseo con validación de calidad de datos"""
        print("⏳ Parseando CSV horario...")
        parsed_data = []
        lineas_error = 0
        
        with open(ruta, 'r', encoding='utf-8') as f:
            for idx, linea in enumerate(f.readlines()[1:], start=2):
                linea = linea.strip()
                if not linea: continue
                
                try:
                    # Limpieza de comillas y comas europeas
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
                    if lineas_error <= 5:  # Solo mostrar primeros 5 errores
                        print(f"  ⚠️  Línea {idx} con error: {str(e)[:50]}")

        print(f"  ✅ {len(parsed_data)} registros parseados, {lineas_error} errores")
        
        df = pd.DataFrame(parsed_data, columns=['FECHA_HORA', 'SECTOR', 'CAUDAL_M3'])
        df['FECHA_HORA'] = pd.to_datetime(df['FECHA_HORA'], format='%d/%m/%Y %H:%M')
        df['Fecha'] = df['FECHA_HORA'].dt.date
        df['Mes'] = df['FECHA_HORA'].dt.month
        df['Hora'] = df['FECHA_HORA'].dt.hour
        df['Año'] = df['FECHA_HORA'].dt.year
        
        # VALIDACIÓN 1: Detectar valores negativos
        negativos = df[df['CAUDAL_M3'] < 0]
        if len(negativos) > 0:
            print(f"  ⚠️  ALERTA: {len(negativos)} registros con caudal negativo detectados")
            df.loc[df['CAUDAL_M3'] < 0, 'CAUDAL_M3'] = 0
        
        # VALIDACIÓN 2: Detectar outliers extremos por sector
        for sector in df['SECTOR'].unique():
            datos_sector = df[df['SECTOR'] == sector]['CAUDAL_M3']
            Q1 = datos_sector.quantile(0.25)
            Q3 = datos_sector.quantile(0.75)
            IQR = Q3 - Q1
            limite_superior = Q3 + 3 * IQR  # 3 IQR (más estricto que 1.5)
            
            outliers = df[(df['SECTOR'] == sector) & (df['CAUDAL_M3'] > limite_superior)]
            if len(outliers) > 0:
                self.anomalias.append({
                    'sector': sector,
                    'outliers': len(outliers),
                    'max_valor': outliers['CAUDAL_M3'].max(),
                    'limite': limite_superior
                })
        
        # VALIDACIÓN 3: Verificar cobertura horaria
        horas_disponibles = sorted(df['Hora'].unique())
        print(f"  📊 Horas disponibles: {horas_disponibles}")
        self.metricas['horas_origen'] = horas_disponibles
        
        return df

    def cargar_datos_mensuales(self, ruta, año_filtro=2024):
        """Carga con filtrado por año para coherencia temporal"""
        print(f"⏳ Cargando CSV mensual (filtrado año {año_filtro})...")
        df_m = pd.read_csv(ruta)
        df_m['Fecha (aaaa/mm/dd)'] = pd.to_datetime(df_m['Fecha (aaaa/mm/dd)'])
        
        # CRÍTICO: Filtrar solo el año que coincida con los datos horarios
        df_m = df_m[df_m['Fecha (aaaa/mm/dd)'].dt.year == año_filtro]
        
        if len(df_m) == 0:
            print(f"  ⚠️  ADVERTENCIA: No hay datos mensuales para {año_filtro}")
            print(f"  📅 Años disponibles: {sorted(pd.read_csv(ruta)['Fecha (aaaa/mm/dd)'].apply(lambda x: x[:4]).unique())}")
            # Continuar con datos disponibles pero documentar discrepancia
            df_m = pd.read_csv(ruta)
            df_m['Fecha (aaaa/mm/dd)'] = pd.to_datetime(df_m['Fecha (aaaa/mm/dd)'])
        else:
            print(f"  ✅ {len(df_m)} registros de facturación en {año_filtro}")
        
        df_m['Mes'] = df_m['Fecha (aaaa/mm/dd)'].dt.month
        df_m['Consumo_M3'] = pd.to_numeric(
            df_m['Consumo (litros)'].astype(str).str.replace(',', ''), 
            errors='coerce'
        ) / 1000.0
        
        # Sumar todos los usos por Barrio y Mes
        df_agg = df_m.groupby(['Barrio', 'Mes'])['Consumo_M3'].sum().reset_index()
        
        self.metricas['consumo_total_mensual'] = df_agg['Consumo_M3'].sum()
        return dict(zip(zip(df_agg.Barrio, df_agg.Mes), df_agg.Consumo_M3))

    def sintetizar_24h(self, df_hora, dict_mensual, mapeo_sectores):
        """Síntesis con múltiples perfiles y validación"""
        print("🧠 Sintetizando horas faltantes...")
        
        # PERFILES DIFERENCIADOS (mejora sobre perfil único)
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
            'MIXTO': {  # Promedio ponderado
                13: 0.08, 14: 0.10, 15: 0.09, 16: 0.07, 17: 0.06, 18: 0.07,
                19: 0.09, 20: 0.11, 21: 0.12, 22: 0.09, 23: 0.07, 0: 0.05
            }
        }
        
        # CLASIFICACIÓN DE SECTORES (mejorable con datos reales)
        TIPO_SECTOR = {
            "CENTRO COMERCIAL GRAN VÍA": 'COMERCIAL',
            "CIUDAD DEPORTIVA DL": 'MIXTO',
            "ALIPARK DL": 'COMERCIAL',
            # Por defecto: MIXTO para sectores no clasificados
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
            usar_salvavidas = False
            metodo_usado = "desconocido"
            
            if barrio is not None:
                total_facturado_barrio = dict_mensual.get((barrio, mes), 0)
                vol_mes_mañana_barrio = vol_barrio_mes.get((barrio, mes), 1e-6)
                
                peso_del_sector = vol_mes_mañana_sector / vol_mes_mañana_barrio if vol_mes_mañana_barrio > 0 else 0
                total_estimado_sector = total_facturado_barrio * peso_del_sector
                vol_faltante_tarde_mes = total_estimado_sector - vol_mes_mañana_sector
                
                if vol_faltante_tarde_mes <= 0:
                    usar_salvavidas = True
                    metodo_usado = "salvavidas_inconsistencia"
                else:
                    sectores_con_mapeo += 1
                    metodo_usado = "facturacion_real"
            else:
                usar_salvavidas = True
                metodo_usado = "salvavidas_sin_mapeo"
                
            if usar_salvavidas:
                sectores_sin_mapeo += 1
                vol_faltante_tarde_mes = (vol_mes_mañana_sector / 0.45) * 0.55

            # MEJORA: Evitar negativos
            vol_faltante_tarde_mes = max(0, vol_faltante_tarde_mes)
            
            ratio_dia = vol_dia_mañana / vol_mes_mañana_sector if vol_mes_mañana_sector > 0 else 0
            vol_faltante_hoy = vol_faltante_tarde_mes * ratio_dia
            
            # MEJORA: Usar perfil específico del sector
            tipo = TIPO_SECTOR.get(sector, 'MIXTO')
            perfil = PERFILES[tipo]
            
            for hora, peso in perfil.items():
                datos_sinteticos.append({
                    'FECHA_HORA': pd.to_datetime(f"{fecha} {hora}:00:00"),
                    'SECTOR': sector,
                    'CAUDAL_M3': round(vol_faltante_hoy * peso, 3),
                    'METODO': metodo_usado
                })

        print(f"  ✅ Sectores con mapeo exitoso: {sectores_con_mapeo}")
        print(f"  ⚠️  Sectores usando salvavidas: {sectores_sin_mapeo}")
        
        df_sintetico = pd.DataFrame(datos_sinteticos)
        df_completo = pd.concat([
            df_hora[['FECHA_HORA', 'SECTOR', 'CAUDAL_M3']].assign(METODO='telemetria_real'),
            df_sintetico
        ])
        df_completo = df_completo.sort_values(['SECTOR', 'FECHA_HORA']).reset_index(drop=True)
        
        # MÉTRICAS DE VALIDACIÓN
        self.metricas['total_original'] = df_hora['CAUDAL_M3'].sum()
        self.metricas['total_sintetico'] = df_sintetico['CAUDAL_M3'].sum()
        self.metricas['ratio_tarde_mañana'] = self.metricas['total_sintetico'] / self.metricas['total_original']
        
        return df_completo

    def generar_reporte_calidad(self):
        """Genera reporte de validación de la síntesis"""
        print("\n" + "="*80)
        print("📊 REPORTE DE CALIDAD DE DATOS")
        print("="*80)
        
        print(f"\n🔢 MÉTRICAS GLOBALES:")
        print(f"  • Volumen telemetría original: {self.metricas['total_original']:,.0f} m³")
        print(f"  • Volumen sintetizado (tarde): {self.metricas['total_sintetico']:,.0f} m³")
        print(f"  • Ratio tarde/mañana: {self.metricas['ratio_tarde_mañana']:.2%}")
        
        if self.metricas['ratio_tarde_mañana'] < 0.5 or self.metricas['ratio_tarde_mañana'] > 2.0:
            print(f"  ⚠️  ALERTA: Ratio fuera del rango esperado (0.5-2.0)")
        
        if self.anomalias:
            print(f"\n⚠️  OUTLIERS DETECTADOS: {len(self.anomalias)} sectores")
            for a in self.anomalias[:5]:  # Top 5
                print(f"  • {a['sector']}: {a['outliers']} valores > {a['limite']:.1f} m³ (max: {a['max_valor']:.1f})")
        
        print("\n✅ Proceso completado")
        print("="*80)


# ==============================================================================
# EJECUCIÓN CON MAPEO
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
    analizador = AnalizadorCaudalMejorado()
    
    df_h = analizador.cargar_datos_horarios(
        '_caudal_medio_sector_hidraulico_hora_2024_-caudal_medio_sector_hidraulico_hora_2024.csv'
    )
    
    dict_m = analizador.cargar_datos_mensuales(
        'datos-hackathon-amaem.xlsx-set-de-datos-.csv',
        año_filtro=2024  # CRÍTICO: usar mismo año
    )
    
    df_final = analizador.sintetizar_24h(df_h, dict_m, MAPEO_SECTORES)
    
    analizador.generar_reporte_calidad()
    
    # Guardar con columna de método para trazabilidad
    df_final.to_csv('Dataset_24H_Mejorado_V2.csv', index=False)
    print("\n💾 Dataset guardado: Dataset_24H_Mejorado_V2.csv")