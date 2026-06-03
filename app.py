"""
=============================================================
  Predictor de Salario - Proyecto Analítica de Datos
  Universidad Icesi | Ingeniería Industrial
=============================================================
  Para correr la app localmente:
      pip install streamlit tensorflow joblib numpy
      streamlit run app.py
=============================================================
"""

import streamlit as st
import numpy as np
import joblib
import os

# ─── Configuración de la página ──────────────────────────────────────────────
st.set_page_config(
    page_title="Predictor de Salario - BN-MLP",
    page_icon="💼",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ─── Estilos CSS personalizados (Premium Dark UI) ───────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
        min-height: 100vh;
    }

    .hero-title {
        font-family: 'Syne', sans-serif;
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #e2b96f, #f0d080, #e2b96f);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        line-height: 1.1;
        margin-bottom: 0.2rem;
    }

    .hero-sub {
        font-family: 'DM Sans', sans-serif;
        font-size: 1rem;
        color: #8892a4;
        text-align: center;
        margin-bottom: 2.5rem;
        letter-spacing: 0.05em;
    }

    .card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(226, 185, 111, 0.15);
        border-radius: 16px;
        padding: 1.8rem 2rem;
        margin-bottom: 1.5rem;
        backdrop-filter: blur(10px);
    }

    .card-title {
        font-family: 'Syne', sans-serif;
        font-size: 0.75rem;
        font-weight: 700;
        color: #e2b96f;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        margin-bottom: 1.2rem;
    }

    .result-box {
        background: linear-gradient(135deg, rgba(226,185,111,0.12), rgba(240,208,128,0.06));
        border: 1px solid rgba(226, 185, 111, 0.4);
        border-radius: 20px;
        padding: 2.5rem;
        text-align: center;
        margin-top: 1.5rem;
    }

    .result-label {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.85rem;
        color: #8892a4;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }

    .result-salary {
        font-family: 'Syne', sans-serif;
        font-size: 3.5rem;
        font-weight: 800;
        color: #e2b96f;
        line-height: 1;
    }

    .result-year {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.9rem;
        color: #8892a4;
        margin-top: 0.4rem;
    }

    .info-badge {
        display: inline-block;
        background: rgba(226,185,111,0.1);
        border: 1px solid rgba(226,185,111,0.2);
        color: #e2b96f;
        border-radius: 20px;
        padding: 0.25rem 0.8rem;
        font-size: 0.78rem;
        font-family: 'DM Sans', sans-serif;
        margin: 0.2rem;
    }

    .warning-box {
        background: rgba(255, 180, 50, 0.08);
        border: 1px solid rgba(255, 180, 50, 0.25);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        color: #f0c060;
        font-size: 0.85rem;
        margin-bottom: 1rem;
    }

    .error-box {
        background: rgba(255, 80, 80, 0.08);
        border: 1px solid rgba(255, 80, 80, 0.25);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        color: #ff8080;
        font-size: 0.85rem;
        margin-bottom: 1rem;
    }

    /* Streamlit UI Overrides */
    .stSelectbox label, .stSlider label, .stNumberInput label {
        color: #b0bcc8 !important;
        font-size: 0.88rem !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    .stSelectbox > div > div {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 10px !important;
        color: #e8edf2 !important;
    }

    .stSlider > div > div > div {
        background: rgba(226,185,111,0.3) !important;
    }

    div[data-testid="stSlider"] > div > div > div > div {
        background: #e2b96f !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #e2b96f, #d4a054) !important;
        color: #0f0f1a !important;
        font-family: 'Syne', sans-serif !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        width: 100% !important;
        letter-spacing: 0.05em !important;
        transition: all 0.2s ease !important;
        cursor: pointer !important;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #f0c870, #e2b96f) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 8px 25px rgba(226,185,111,0.3) !important;
    }

    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─── Cargar modelo y artefactos del BN-MLP ────────────────────────────────────
@st.cache_resource
def cargar_modelo():
    """Carga de forma eficiente la arquitectura entrenada con BatchNorm."""
    archivos = {
        'modelo':   'modelo_salario.keras',
        'scaler':   'scaler_salario.pkl',
        'encoders': 'encoders_salario.pkl'
    }

    faltantes = [nombre for nombre, archivo in archivos.items() if not os.path.exists(archivo)]

    if faltantes:
        return None, None, None, faltantes

    try:
        import tensorflow as tf
        modelo   = tf.keras.models.load_model(archivos['modelo'])
        scaler   = joblib.load(archivos['scaler'])
        encoders = joblib.load(archivos['encoders'])
        return modelo, scaler, encoders, []
    except Exception as e:
        return None, None, None, [str(e)]


modelo, scaler, encoders, errores = cargar_modelo()


# ─── Listas de categorías (Fallback por si no cargan los encoders) ───────────
WORK_TYPES      = ['Contract', 'Full-Time', 'Intern', 'Part-Time', 'Temporary']
QUALIFICATIONS  = ['BA', 'BBA', 'BCA', 'B.Com', 'B.Tech', 'MBA', 'MCA', 'M.Com', 'M.Tech', 'PhD']
COUNTRIES_TOP   = [
    'Afghanistan', 'Albania', 'Algeria', 'Argentina', 'Australia',
    'Austria', 'Belgium', 'Brazil', 'Canada', 'Chile', 'China',
    'Colombia', 'Czech Republic', 'Denmark', 'Egypt', 'Finland',
    'France', 'Germany', 'Greece', 'Hungary', 'India', 'Indonesia',
    'Iran', 'Iraq', 'Ireland', 'Israel', 'Italy', 'Japan', 'Jordan',
    'Kenya', 'Malaysia', 'Mexico', 'Morocco', 'Netherlands',
    'New Zealand', 'Nigeria', 'Norway', 'Pakistan', 'Peru',
    'Philippines', 'Poland', 'Portugal', 'Romania', 'Russia',
    'Saudi Arabia', 'South Africa', 'South Korea', 'Spain', 'Sweden',
    'Switzerland', 'Thailand', 'Turkey', 'Ukraine', 'United Kingdom',
    'United States', 'Venezuela', 'Vietnam'
]
ROLES_TOP = [
    'Account Manager', 'Business Analyst', 'Cloud Architect',
    'Customer Success Manager', 'Data Analyst', 'Data Engineer',
    'Data Scientist', 'DevOps Engineer', 'Financial Analyst',
    'Frontend Developer', 'Full Stack Developer', 'HR Manager',
    'Interaction Designer', 'Marketing Manager', 'Network Administrator',
    'Operations Manager', 'Product Manager', 'Project Manager',
    'Sales Manager', 'Software Engineer', 'System Administrator',
    'UI/UX Designer', 'User Interface Designer'
]

def get_clases(nombre, fallback):
    """Extrae las clases reales guardadas en el LabelEncoder."""
    if encoders and nombre in encoders:
        return list(encoders[nombre].classes_)
    return fallback


# ─── Lógica de Procesamiento y Predicción ─────────────────────────────────────
def predecir_salario(experience, company_size, latitude, longitude,
                     year, work_type, qualification, country, role):
    """Preprocesa las entradas, aplica escalamiento y ejecuta la red neuronal."""

    def encode(col_name, valor, fallback_list):
        if encoders and col_name in encoders:
            le = encoders[col_name]
            if valor in le.classes_:
                return int(le.transform([valor])[0])
            else:
                return len(le.classes_) // 2
        else:
            if valor in fallback_list:
                return fallback_list.index(valor)
            return 0

    wt_enc   = encode('Work Type',      work_type,     WORK_TYPES)
    qual_enc = encode('Qualifications', qualification, QUALIFICATIONS)
    co_enc   = encode('Country',        country,       COUNTRIES_TOP)
    role_enc = encode('Role',           role,          ROLES_TOP)

    # Orden exacto del vector de entrenamiento (9 características)
    X_input = np.array([[
        experience, company_size, latitude, longitude,
        year, wt_enc, qual_enc, co_enc, role_enc
    ]])

    # Escalar numéricas y categóricas según el pipeline neuronal
    X_scaled = scaler.transform(X_input)

    # Inferencia con BN-MLP
    pred = modelo.predict(X_scaled, verbose=0)
    return float(pred[0][0])


# ═══════════════════════════════════════════════════════════════════════════════
#  DISEÑO DE LA INTERFAZ DE USUARIO
# ═══════════════════════════════════════════════════════════════════════════════

# Sección de Cabecera (Hero)
st.markdown('<p class="hero-title">💼 Salary Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Ingresa las características del cargo para estimar el salario promedio usando la red neuronal BN-MLP</p>', unsafe_allow_html=True)

# Alerta flotante en caso de ausencia de archivos
if errores:
    st.markdown(f"""
    <div class="warning-box">
        ⚠️ <strong>Modelo o artefactos no detectados.</strong><br>
        Asegúrate de que los archivos exportados (<code>modelo_salario.keras</code>, 
        <code>scaler_salario.pkl</code> y <code>encoders_salario.pkl</code>) 
        estén cargados directamente en la raíz de tu repositorio de GitHub.<br><br>
        <em>Detalle técnico: {', '.join(errores)}</em>
    </div>
    """, unsafe_allow_html=True)

# ── Tarjeta 1: Parámetros del Perfil y la Empresa ────────────────────────────
st.markdown('<div class="card"><p class="card-title">🏢 Experiencia & Empresa</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    experience = st.slider(
        "Años de experiencia requeridos",
        min_value=0, max_value=20, value=3, step=1,
        help="Experiencia laboral mínima solicitada por la vacante"
    )
with col2:
    company_size = st.number_input(
        "Tamaño de la empresa (N° empleados)",
        min_value=10, max_value=500000, value=5000, step=500,
        help="Volumen total de empleados dentro de la organización corporativa"
    )

col3, col4 = st.columns(2)
with col3:
    qualification = st.selectbox(
        "Nivel educativo exigido",
        options=get_clases('Qualifications', QUALIFICATIONS)
    )
with col4:
    work_type = st.selectbox(
        "Tipo de jornada laboral",
        options=get_clases('Work Type', WORK_TYPES)
    )

st.markdown('</div>', unsafe_allow_html=True)

# ── Tarjeta 2: Atributos del Cargo y Ubicación Geográfica ──────────────────────
st.markdown('<div class="card"><p class="card-title">📍 Cargo & Ubicación</p>', unsafe_allow_html=True)

role = st.selectbox(
    "Rol profesional / Cargo",
    options=get_clases('Role', ROLES_TOP)
)

col5, col6 = st.columns(2)
with col5:
    country = st.selectbox(
        "País de contratación",
        options=get_clases('Country', COUNTRIES_TOP)
    )
with col6:
    year = st.selectbox(
        "Año de publicación de la oferta",
        options=list(range(2019, 2027)),
        index=4
    )

col7, col8 = st.columns(2)
with col7:
    latitude = st.number_input(
        "Latitud geográfica",
        min_value=-90.0, max_value=90.0, value=4.7110,
        format="%.4f",
        help="Coordenada decimal norte-sur del lugar de empleo"
    )
with col8:
    longitude = st.number_input(
        "Longitud geográfica",
        min_value=-180.0, max_value=180.0, value=-74.0721,
        format="%.4f",
        help="Coordenada decimal este-oeste del lugar de empleo"
    )

st.markdown('</div>', unsafe_allow_html=True)

# ── Ejecución del Algoritmo de Inferencia ─────────────────────────────────────
predict_btn = st.button("🔮 Calcular Salario Estimado")

if predict_btn:
    if modelo is None:
        st.markdown("""
        <div class="error-box">
            ❌ <strong>Error de Ejecución:</strong> No se puede procesar la inferencia debido a que los pesos de la red neuronal 
            no han sido cargados. Comprueba las rutas del repositorio.
        </div>
        """, unsafe_allow_html=True)
    else:
        with st.spinner("Procesando datos a través de las capas BN-MLP..."):
            try:
                # Calcular predicción anual
                salario = predecir_salario(
                    experience, company_size, latitude, longitude,
                    year, work_type, qualification, country, role
                )

                salario_mes = salario / 12

                # Renderizado estético del contenedor de resultados
                st.markdown(f"""
                <div class="result-box">
                    <p class="result-label">Salario Anual Estimado (Predicción Red Neural)</p>
                    <p class="result-salary">${salario:,.0f}</p>
                    <p class="result-year">≈ ${salario_mes:,.0f} / mes &nbsp;·&nbsp; USD</p>
                </div>
                """, unsafe_allow_html=True)

                # Badges dinámicos con resumen de los datos ingresados
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"""
                <div style="text-align:center; margin-top:0.5rem;">
                    <span class="info-badge">🎓 {qualification}</span>
                    <span class="info-badge">💼 {work_type}</span>
                    <span class="info-badge">📌 {role}</span>
                    <span class="info-badge">🌍 {country}</span>
                    <span class="info-badge">⏱ {experience} Años Exp.</span>
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.markdown(f"""
                <div class="error-box">
                    ❌ <strong>Error de Inferencia:</strong> Ocurrió una anomalía matemática durante la transformación: <code>{str(e)}</code>
                </div>
                """, unsafe_allow_html=True)

# ── Bloque Informativo / Créditos Academicos ──────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#4a5568; font-size:0.78rem; margin-top:2rem; font-family:'DM Sans',sans-serif;">
    Proyecto Analítica de Datos · Universidad Icesi · 2026<br>
    Angely Palomeque · Gabriela Bolaños · Mariana Jiménez · Haider Basante
</div>
""", unsafe_allow_html=True)