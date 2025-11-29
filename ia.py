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
st.set_page_config(page_title="AI Soccer NASA", layout="wide", page_icon="🚀")
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
# 2. INGENIERÍA DE DATOS (EL CEREBRO NUEVO) 🧠
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
        df = df.sort_values('date') # Importante ordenar por fecha
        
        # TARGET: 0=Draw, 1=Home, 2=Away
        conditions = [
            (df['home_goals'] > df['away_goals']),
            (df['home_goals'] < df['away_goals'])
        ]
        df['result'] = np.select(conditions, [1, 2], default=0)
        
        return df
    except: return pd.DataFrame()

def calculate_rolling_features(df):
    """
    Crea estadísticas de forma reciente (últimos 3 partidos) para cada equipo.
    Esta es la parte difícil: convertir formato Local-Visitante a formato Equipo-Partido.
    """
    # 1. Creamos un DataFrame vertical (Un equipo por fila)
    home_df = df[['date', 'home', 'home_goals', 'away_goals', 'result']].copy()
    home_df.columns = ['date', 'team', 'gf', 'ga', 'res_match']
    home_df['points'] = np.where(home_df['res_match']==1, 3, np.where(home_df['res_match']==0, 1, 0))
    home_df['is_home'] = 1

    away_df = df[['date', 'away', 'away_goals', 'home_goals', 'result']].copy()
    away_df.columns = ['date', 'team', 'gf', 'ga', 'res_match']
    away_df['points'] = np.where(away_df['res_match']==2, 3, np.where(away_df['res_match']==0, 1, 0))
    away_df['is_home'] = 0

    # 2. Concatenamos y ordenamos
    team_stats = pd.concat([home_df, away_df]).sort_values(['team', 'date'])

    # 3. Calculamos Rolling Averages (Últimos 3 partidos)
    # .shift() es vital: No podemos usar el partido de HOY para predecir HOY.
    window = 3
    team_stats['form_points'] = team_stats.groupby('team')['points'].transform(lambda x: x.rolling(window, min_periods=1).sum().shift(1))
    team_stats['form_gf'] = team_stats.groupby('team')['gf'].transform(lambda x: x.rolling(window, min_periods=1).mean().shift(1))
    team_stats['form_ga'] = team_stats.groupby('team')['ga'].transform(lambda x: x.rolling(window, min_periods=1).mean().shift(1))

    # Rellenamos NaN (primeros partidos de liga) con 0
    team_stats = team_stats.fillna(0)

    # 4. Volvemos a pegar estos datos en el DataFrame original (Join)
    # Unimos datos del LOCAL
    df = df.merge(team_stats[['date', 'team', 'form_points', 'form_gf', 'form_ga']], 
                  left_on=['date', 'home'], right_on=['date', 'team'], how='left')
    df = df.rename(columns={'form_points': 'h_form', 'form_gf': 'h_att_recent', 'form_ga': 'h_def_recent'})
    df = df.drop(columns=['team'])

    # Unimos datos del VISITANTE
    df = df.merge(team_stats[['date', 'team', 'form_points', 'form_gf', 'form_ga']], 
                  left_on=['date', 'away'], right_on=['date', 'team'], how='left')
    df = df.rename(columns={'form_points': 'a_form', 'form_gf': 'a_att_recent', 'form_ga': 'a_def_recent'})
    df = df.drop(columns=['team'])

    return df, team_stats # Retornamos también team_stats para buscar info actual

# ======================================================
# 3. MODELO 1: DIXON-COLES
# ======================================================
def calculate_strengths(df):
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
    pd_prob = np.diag(probs).sum()
    pa = np.triu(probs, 1).sum()
    return ph, pd_prob, pa, h_exp, a_exp

# ======================================================
# 4. MODELO 2: RANDOM FOREST AVANZADO (FORMA RECIENTE) 🤖
# ======================================================
def train_ai_model(df):
    """Entrena Random Forest con Historial + Forma Reciente"""
    le = LabelEncoder()
    all_teams = pd.concat([df['home'], df['away']]).unique()
    le.fit(all_teams)
    
    df['home_code'] = le.transform(df['home'])
    df['away_code'] = le.transform(df['away'])
    
    # FEATURES AVANZADAS:
    # IDs + Cuotas + Forma Local (Puntos, Ataque, Defensa) + Forma Visita
    features = ['home_code', 'away_code', 'odd_h', 'odd_d', 'odd_a',
                'h_form', 'h_att_recent', 'h_def_recent',
                'a_form', 'a_att_recent', 'a_def_recent']
    
    X = df[features]
    y = df['result']
    
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X, y)
    
    return model, le

def get_latest_team_stats(team, team_stats_df):
    """Busca los datos más recientes de un equipo para predecir el futuro"""
    team_data = team_stats_df[team_stats_df['team'] == team].tail(1)
    if team_data.empty: return 0, 0, 0 # Sin datos
    # Para el futuro, su 'Rolling' incluye el último partido jugado
    # Simplemente recalculamos rápido el rolling de los últimos 3
    last_3 = team_stats_df[team_stats_df['team'] == team].tail(3)
    form = last_3['points'].sum()
    att = last_3['gf'].mean()
    deff = last_3['ga'].mean()
    return form, att, deff

def predict_ai_advanced(model, le, team_stats_df, home, away, odd_h, odd_d, odd_a):
    try:
        h_code = le.transform([home])[0]
        a_code = le.transform([away])[0]
        
        # Obtener forma reciente REAL
        h_form, h_att, h_def = get_latest_team_stats(home, team_stats_df)
        a_form, a_att, a_def = get_latest_team_stats(away, team_stats_df)
        
        input_data = pd.DataFrame([[h_code, a_code, odd_h, odd_d, odd_a,
                                    h_form, h_att, h_def, a_form, a_att, a_def]], 
                                columns=['home_code', 'away_code', 'odd_h', 'odd_d', 'odd_a',
                                         'h_form', 'h_att_recent', 'h_def_recent',
                                         'a_form', 'a_att_recent', 'a_def_recent'])
        
        probs = model.predict_proba(input_data)[0]
        
        classes = model.classes_
        p_draw, p_home, p_away = 0.0, 0.0, 0.0
        for i, class_val in enumerate(classes):
            if class_val == 0: p_draw = probs[i]
            if class_val == 1: p_home = probs[i]
            if class_val == 2: p_away = probs[i]
            
        return p_home, p_draw, p_away, h_form, a_form
        
    except Exception as e:
        return 0.33, 0.33, 0.34, 0, 0

# ======================================================
# 5. INTERFAZ GRÁFICA
# ======================================================
with st.sidebar:
    st.header("⚙️ Configuración")
    leagues = {"SP1": "🇪🇸 La Liga", "E0": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League", "I1": "🇮🇹 Serie A", "D1": "🇩🇪 Bundesliga"}
    code = st.selectbox("Liga", list(leagues.keys()), format_func=lambda x: leagues[x])
    
    raw_df = fetch_live_soccer_data(code)
    if raw_df.empty: st.error("Error datos"); st.stop()
    
    # PROCESAMIENTO
    df_processed, team_stats_history = calculate_rolling_features(raw_df)
    
    st.divider()
    model_type = st.radio("Cerebro:", ["Dixon-Coles (Stats)", "Random Forest (IA Avanzada)"], index=1)
    
    if "IA" in model_type:
        ai_model, encoder = train_ai_model(df_processed)
        st.success(f"🚀 IA Entrenada con {len(df_processed)} partidos + Forma Reciente")
    else:
        stats, ah, aa = calculate_strengths(raw_df)
        st.success("📐 Modelo Estadístico Listo")

st.title(f"⚽ {leagues[code]} - {model_type}")

teams = sorted(raw_df['home'].unique())
c1, c2 = st.columns(2)
home = c1.selectbox("Local", teams, index=0)
away = c2.selectbox("Visitante", [t for t in teams if t != home], index=0)

st.info("ℹ️ Ingresa cuotas reales para activar la predicción")
co1, co2, co3 = st.columns(3)
odd_h = co1.number_input("Cuota 1", 1.01, 20.0, 2.0)
odd_d = co2.number_input("Cuota X", 1.01, 20.0, 3.2)
odd_a = co3.number_input("Cuota 2", 1.01, 20.0, 3.5)

# --- EJECUCIÓN ---
h_points_recent, a_points_recent = 0, 0

if "IA" in model_type:
    ph, pd_prob, pa, h_points_recent, a_points_recent = predict_ai_advanced(
        ai_model, encoder, team_stats_history, home, away, odd_h, odd_d, odd_a)
    h_exp, a_exp = 0, 0
    model_msg = "IA analizando cuotas + Puntos últimos 3 partidos"
else:
    ph, pd_prob, pa, h_exp, a_exp = predict_dixon_coles(home, away, stats, ah, aa)
    model_msg = "Modelo estadístico puro"

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

if "IA" in model_type:
    st.caption(f"📊 Datos usados por la IA: {home} ({h_points_recent}/9 ptos recientes) vs {away} ({a_points_recent}/9 ptos recientes)")

# Valor Esperado
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
