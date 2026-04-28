# 💧 AquaAlert: Gemelo Digital Predictivo Híbrido para Redes de Abastecimiento

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Machine Learning](https://img.shields.io/badge/Machine_Learning-XGBoost-orange.svg)
![GenAI](https://img.shields.io/badge/GenAI-Llama_3.3-green.svg)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)

> **Proyección de estrés hídrico urbano integrando series temporales inerciales y contexto sociológico en tiempo real (OSINT + APIs).**

![AquaAlert Dashboard](https://img.shields.io/badge/Insertar_Captura_Dashboard_Aqui-Gris)

## 📌 Visión General del Proyecto

**AquaAlert** es una solución End-to-End de Machine Learning diseñada para anticipar anomalías en la demanda de agua en los 43 sectores hidráulicos de la ciudad de Alicante. 

El proyecto resuelve una limitación histórica en la gestión de infraestructuras críticas: **la ceguera de los modelos estadísticos tradicionales ante el comportamiento humano**. Un modelo clásico no sabe si hoy es festivo, si hay una ola de calor extrema o si ha atracado un megacrucero con 5.000 pasajeros. Para resolver esto, he diseñado una **Arquitectura de Ensamble Híbrido** que separa la "física" de la red (inercia histórica) de la "sociología" de la ciudad (contexto en tiempo real).

## 🧠 Arquitectura del Sistema

El pipeline se divide en tres capas secuenciales e independientes:

### 1. Data Engineering & Imputación (`src/Síntesis.py`)
* **Homogeneización de Datos Asimétricos:** Cruce de telemetría horaria con datos de facturación mensual para sintetizar un *Ground Truth* de 24 horas continuas sin valores nulos.
* **Perfiles Estocásticos:** Imputación de horas faltantes (Data Imputation) aplicando pesos de consumo según la tipología del sector (Residencial, Comercial, Industrial, Mixto).

### 2. Capa Sensorial GenAI / Zero-Shot Feature Extraction (`src/agente_llm.py`)
* **Transductor de Contexto:** Integración de **Llama-3.3 70B** (vía Groq) para analizar en tiempo real 11 fuentes de datos abiertas:
  * Climatología (Open-Meteo).
  * Tráfico Aéreo (OpenSky Network) y Portuario (Cruisewatch Web Scraping).
  * Ocupación Hotelera (INE) y Movilidad (MITMA).
  * Eventos Multitudinarios (Ticketmaster API + Histórico local).
  * Escucha Social OSINT (Prensa RSS y Foros de Reddit).
* **Generación de Tensores:** La IA procesa la información y exporta un JSON estricto con factores multiplicativos calibrados, actuando como un extractor de *features* dinámico.

### 3. Motor Predictivo Híbrido (`src/Conjunto.py`)
* **Machine Learning Clásico:** Un modelo **XGBoost Regressor** validado mediante `TimeSeriesSplit` (5 folds) calcula el caudal base inercial utilizando retardos (Lags 24h/168h) y variables cíclicas.
* **Fusión Espaciotemporal:** Un motor de reglas Python recibe la predicción base (ML) y la ajusta aplicando los tensores sociológicos (GenAI) mediante **media ponderada**.
* **Clipping y Enrutamiento:** Los tensores se aplican estrictamente a las zonas afectadas (ej. los cruceros solo alteran la Zona Centro) y están limitados matemáticamente al rango `[0.70, 1.50]` para evitar explosiones algorítmicas y garantizar límites físicos.

## 📂 Estructura del Repositorio

El proyecto sigue los estándares de organización de *Cookiecutter Data Science*:

```text
AquaAlert/
│
├── data/                      # 🗄️ Entorno de Datos
│   ├── raw/                   # Datasets originales inmutables
│   └── processed/             # Datasets limpios tras Data Engineering
│
├── outputs/                   # 📊 Resultados, JSONs y logs generados
│
├── mapas/                     # 🗺️ Topología y GeoJSON de la red hídrica
│
├── src/                       # 💻 Código Fuente (Lógica Core)
│   ├── Síntesis.py            # Pipeline de ingesta e imputación
│   ├── agente_llm.py          # Extracción OSINT y agente LLM
│   └── Conjunto.py            # Entrenamiento XGBoost y Ensamble
│
├── app.py                     # 🖥️ Dashboard interactivo (Streamlit)
├── requirements.txt           # 📦 Dependencias del entorno
├── .env.example               # 🔑 Plantilla de variables de entorno
└── README.md                  # 📖 Documentación del proyecto
💻 Tech Stack
Core ML & Data: Pandas, NumPy, Scikit-Learn, XGBoost.

GenAI & Integraciones: Groq SDK (Llama-3), BeautifulSoup4 (Scraping), Requests.

Visualización Espacial: Folium, Streamlit-Folium, Plotly, Pyproj.

Despliegue & MLOps: Streamlit, Git, Entornos Virtuales.

🚀 Reproducibilidad y Ejecución
El proyecto está diseñado para ejecutarse localmente sin configuraciones complejas.

1. Clonar e Instalar
Bash
git clone [https://github.com/](https://github.com/)[TuUsuario]/AquaAlert.git
cd AquaAlert
pip install -r requirements.txt
2. Configuración de Entorno
Crea un archivo .env en la raíz del proyecto y añade tus claves de API:

Plaintext
GROQ_API_KEY="tu_clave_de_groq_aqui"
TICKETMASTER_KEY="tu_clave_de_ticketmaster_aqui"
3. Ejecución del Pipeline Analítico
Para procesar los datos, consultar el entorno sociológico y generar las predicciones, ejecuta el motor principal (este script orquesta automáticamente el resto):

Bash
python src/Conjunto.py
4. Lanzar el Dashboard (Interfaz de Usuario)
Una vez generada la predicción en la carpeta /outputs, levanta la interfaz gráfica:

Bash
streamlit run app.py
(Nota de Resiliencia: Si el sistema se ejecuta sin acceso a Internet o sin API Keys, el agente LLM aplicará automáticamente un mecanismo de "Fallback", utilizando priors estacionales puramente matemáticos para que la predicción nunca se interrumpa).

📬 Contacto & Portfolio
Jesús Ortiz - Estudiante de Ingeniería Biomédica

💼 LinkedIn: www.linkedin.com/in/jesús-ortiz-74857331a

🐙 GitHub: https://github.com/Jesus-OP

📧 Email: jesus.ortiz.perona@gmail.com