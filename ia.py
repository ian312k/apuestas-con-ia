import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import plotly.graph_objects as go
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ======================================================
# 1. CONFIGURACIÓN
# ======================================================
st.set_page_config(page_title="AI Soccer Analyst", layout="wide", page_icon="🤖")
CSV_FILE = 'mis_apuestas_ml.csv'

st.markdown("""
<style>
    div[data-testid="stMetric"] {
        background-color: #262730;
        border: 1px solid #464b5c;
        padding: 15px;
        border-radius: 10px;
    }
    h1, h2, h3 { text-align: center; }
</style>
""", unsafe_allow_html=True)

# ======================================================
# 2. CARGA DE DATOS
# ======================================================
@st.cache_data
def fetch_live_soccer_data(league_code="SP1"):
    url = f"https://www.football-data.co.uk/mmz4281/2526/{league_code}.csv"
    try:
        df = pd.read_csv(url)
        cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'B365H', 'B365D', 'B365A', 'HST', 'AST']
        actual = [c for c in cols if c in df.columns]
        df = df[actual]
        
        mapping = {
            'Date': 'date', 'HomeTeam': 'home', 'AwayTeam': 'away', 
            'FTHG': 'home_goals', 'FTAG': 'away_goals', 
            'B365H': 'odd_h', 'B365D': 'odd_d', 'B365A': 'odd_a',
            'HST': 'home_shots', 'AST': 'away_shots'
        }
        df = df.rename(columns=mapping).dropna()
        df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
        
        # CREAMOS LA COLUMNA "RESULTADO" (TARGET) PARA QUE LA IA APRENDA
        # 0 = Empate, 1 = Local, 2 = Visitante
        conditions = [
            (df['home_goals'] > df['away_goals']),
            (df['home_goals'] < df['away_goals'])
        ]
        choices = [1, 2] # 1: Home Win, 2: Away Win
        df['result'] = np.select(conditions, choices, default=0) # 0: Draw
        
        return df
    except: return pd.DataFrame()

# ======================================================
# 3. MODELO 1: DIXON-COLES (ESTADÍSTICO)
# ======================================================
def calculate_strengths(df):
    # ... (Tu código de Dixon-Coles original se mantiene aquí para cuando elijas ese modo) ...
    last_date = df['date'].max()
    df['days_ago'] = (last_date - df['date']).dt.days
    df['weight'] = np.exp(-0.005 * df['days_ago'])
    
    avg_h = np.average(df['home_goals'], weights=df['weight'])
    avg_a = np.average(df['away_goals'], weights=df['weight'])
    
    stats = {}
    for team in set(df['home'].unique()) | set(df['away'].unique()):
        h_m = df[df['home'] == team]
        a_m = df[df['away'] == team]
        
        att_h = np.average(h_m['home_goals'], weights=h_m['weight'])/avg_h if not h_m.empty else 1.0
        def_h = np.average(h_m['away_goals'], weights=h_m['weight'])/avg_a if not h_m.empty else 1.0
        att_a = np.average(a_m['away_goals'], weights=a_m['weight'])/avg_a if not a_m.empty else 1.0
        def_a = np.average(a_m['home_goals'], weights=a_m['weight'])/avg_h if not a_m.empty else 1.0
        
        stats[team] = {'att_h': att_h, 'def_h': def_h, 'att_a': att_a, 'def_a': def_a}
    return stats, avg_h, avg_a

def predict_dixon_coles(home, away, stats, avg_h, avg_a):
    h_exp = stats[home]['att_h'] * stats[away]['def_a'] * avg_h
    a_exp = stats[away]['att_a'] * stats[home]['def_h'] * avg_a
    max_g = 10
    probs = np.zeros((max_g, max_g))
    rho = -0.13
    
    for x in range(max_g):
        for y in range(max_g):
            p = poisson.pmf(x, h_exp) * poisson.pmf(y, a_exp)
            if x==0 and y==0: c = 1 - (h_exp*a_exp*rho)
            elif x==0 and y==1: c = 1 + (h_exp*rho)
            elif x==1 and y==0: c = 1 + (a_exp*rho)
            elif x==1 and y==1: c = 1 - rho
            else: c = 1.0
            probs[x][y] = p * c
            
    probs = np.maximum(0, probs)
    probs = probs / probs.sum()
    
    ph = np.tril(probs, -1).sum()
    pd = np.diag(probs).sum()
    pa = np.triu(probs, 1).sum()
    return ph, pd, pa, h_exp, a_exp

# ======================================================
# 4. MODELO 2: RANDOM FOREST (MACHINE LEARNING) 🤖
# ======================================================
def train_ai_model(df):
    """Entrena una IA para predecir el ganador basándose en Cuotas y Equipos"""
    # 1. Preparamos los datos
    # La IA no entiende nombres como "Barcelona", hay que convertirlos a números
    le = LabelEncoder()
    all_teams = pd.concat([df['home'], df['away']]).unique()
    le.fit(all_teams)
    
    df['home_code'] = le.transform(df['home'])
    df['away_code'] = le.transform(df['away'])
    
    # 2. Definimos las "Features" (Qué datos mira la IA)
    # Mira: Quién juega (codificado) y qué dicen las casas de apuestas (Cuotas)
    features = ['home_code', 'away_code', 'odd_h', 'odd_d', 'odd_a']
    X = df[features]
    y = df['result'] # Lo que queremos predecir (0, 1, 2)
    
    # 3. Entrenamos el Cerebro (Random Forest)
    # n_estimators=100 significa que crea 100 árboles de decisión mentales
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model, le

def predict_ai(model, le, home, away, odd_h, odd_d, odd_a):
    """Usa la IA entrenada para predecir un partido futuro"""
    try:
        # Preparamos los datos del partido futuro igual que los de entrenamiento
        h_code = le.transform([home])[0]
        a_code = le.transform([away])[0]
        
        # La IA necesita ver los datos en el mismo orden
        input_data = pd.DataFrame([[h_code, a_code, odd_h, odd_d, odd_a]], 
                                columns=['home_code', 'away_code', 'odd_h', 'odd_d', 'odd_a'])
        
        # .predict_proba() nos da los porcentajes [Prob_Empate, Prob_Local, Prob_Visita]
        probs = model.predict_proba(input_data)[0]
        
        # Random Forest a veces devuelve el orden distinto, aseguramos:
        # Las clases suelen ser ordenadas: 0 (Draw), 1 (Home), 2 (Away)
        # Nota: Sklearn ordena las clases numéricamente.
        
        # Mapeo seguro
        classes = model.classes_
        p_draw, p_home, p_away = 0.0, 0.0, 0.0
        
        for i, class_val in enumerate(classes):
            if class_val == 0: p_draw = probs[i]
            if class_val == 1: p_home = probs[i]
            if class_val == 2: p_away = probs[i]
            
        return p_home, p_draw, p_away
        
    except Exception as e:
        return 0.33, 0.33, 0.34 # Fallback si hay error (equipos nuevos)

# ======================================================
# 5. INTERFAZ GRÁFICA
# ======================================================
with st.sidebar:
    st.header("⚙️ Configuración")
    leagues = {"SP1": "🇪🇸 La Liga", "E0": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League", "I1": "🇮🇹 Serie A", "D1": "🇩🇪 Bundesliga"}
    code = st.selectbox("Liga", list(leagues.keys()), format_func=lambda x: leagues[x])
    
    df = fetch_live_soccer_data(code)
    if df.empty: st.error("Error datos"); st.stop()
    
    # --- SELECTOR DE CEREBRO ---
    st.divider()
    st.markdown("### 🧠 Selecciona tu Inteligencia")
    model_type = st.radio("Modelo:", ["Dixon-Coles (Matemático)", "Random Forest (IA)"], index=0)
    
    # Entrenar modelos según selección
    if "IA" in model_type:
        ai_model, encoder = train_ai_model(df)
        st.success(f"🤖 IA Entrenada con {len(df)} partidos")
    else:
        stats, ah, aa = calculate_strengths(df)
        st.success(f"📐 Estadística calculada")

st.title(f"⚽ {leagues[code]} - {model_type}")

# Selección de equipos
teams = sorted(pd.concat([df['home'], df['away']]).unique())
c1, c2 = st.columns(2)
home = c1.selectbox("Local", teams, index=0)
away = c2.selectbox("Visitante", [t for t in teams if t != home], index=0)

# INPUTS DE CUOTAS (Necesarios para la IA)
st.info("ℹ️ Para la IA, las cuotas son vitales. Introduce las cuotas reales de hoy.")
co1, co2, co3 = st.columns(3)
odd_h = co1.number_input("Cuota Local", 1.01, 20.0, 2.0)
odd_d = co2.number_input("Cuota Empate", 1.01, 20.0, 3.2)
odd_a = co3.number_input("Cuota Visita", 1.01, 20.0, 3.5)

# --- EJECUCIÓN DEL CEREBRO ---
if "IA" in model_type:
    # Predicción Machine Learning
    ph, pd_prob, pa = predict_ai(ai_model, encoder, home, away, odd_h, odd_d, odd_a)
    h_exp, a_exp = 0, 0 # La IA Random Forest no calcula goles esperados directamente
    msg_model = "Basado en patrones de apuestas y resultados previos."
else:
    # Predicción Dixon-Coles
    ph, pd_prob, pa, h_exp, a_exp = predict_dixon_coles(home, away, stats, ah, aa)
    msg_model = "Basado en fuerza de ataque y defensa."

# --- VISUALIZACIÓN ---
st.divider()
g1, g2, g3 = st.columns(3)

def plot_gauge(val, title, color):
    return go.Figure(go.Indicator(
        mode="gauge+number", value=val*100, title={'text': title},
        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': color}, 'bgcolor': "white"}
    )).update_layout(height=150, margin=dict(l=20, r=20, t=30, b=20))

g1.plotly_chart(plot_gauge(ph, f"Gana {home}", "#4CAF50"), use_container_width=True)
g2.plotly_chart(plot_gauge(pd_prob, "Empate", "#FFC107"), use_container_width=True)
g3.plotly_chart(plot_gauge(pa, f"Gana {away}", "#2196F3"), use_container_width=True)

st.caption(f"🧠 Lógica: {msg_model}")

# Valor Esperado (EV)
ev_h = (ph * odd_h) - 1
ev_d = (pd_prob * odd_d) - 1
ev_a = (pa * odd_a) - 1

st.markdown("### 🏦 Análisis de Valor")
v1, v2, v3 = st.columns(3)
def show_val(label, ev):
    if ev > 0: st.success(f"{label}: +{ev*100:.1f}% EV ✅")
    else: st.error(f"{label}: {ev*100:.1f}% EV ❌")

v1.show_val(f"Apostar {home}", ev_h)
v2.show_val("Apostar Empate", ev_d)
v3.show_val(f"Apostar {away}", ev_a)