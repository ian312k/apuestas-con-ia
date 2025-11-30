import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import plotly.graph_objects as go
import os
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ======================================================
# 1. CONFIGURACIÓN Y ESTILOS CSS (DARK MODE) 🎨
# ======================================================
st.set_page_config(page_title="Predicctor de Fútbol IA + Pro", layout="wide", page_icon="⚽")
CSV_FILE = 'mis_apuestas_pro.csv'

# Inicializar Session State
if 'ticket' not in st.session_state:
    st.session_state.ticket = []

# Estilos CSS
st.markdown("""
<style>
    div[data-testid="stMetric"] {
        background-color: #262730;
        border: 1px solid #464b5c;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
    }
    .ticket-box {
        background-color: #1e1e1e;
        border: 1px solid #ffd700;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    h1, h2, h3 { text-align: center; }
</style>
""", unsafe_allow_html=True)

# ======================================================
# 2. LÓGICA DE DATOS Y FEATURE ENGINEERING 🧠
# ======================================================
@st.cache_data
def fetch_live_soccer_data(league_code="SP1"):
    """Descarga datos de football-data.co.uk"""
    url = f"https://www.football-data.co.uk/mmz4281/2526/{league_code}.csv"
    try:
        df = pd.read_csv(url)
        cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'B365H', 'B365D', 'B365A']
        actual_cols = [c for c in cols if c in df.columns]
        df = df[actual_cols]
        
        new_names = ['date', 'home', 'away', 'home_goals', 'away_goals', 'odd_h', 'odd_d', 'odd_a']
        if len(actual_cols) == 8:
            df.columns = new_names
        else:
            # Fallback si no hay cuotas
            df = df[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']]
            df.columns = ['date', 'home', 'away', 'home_goals', 'away_goals']
            df['odd_h'] = 1.0; df['odd_d'] = 1.0; df['odd_a'] = 1.0 

        df = df.dropna()
        df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
        return df.sort_values('date')
    except: return pd.DataFrame()

# --- A. LÓGICA DIXON-COLES (Estadística) ---
def calculate_strengths(df):
    last_date = df['date'].max()
    df['days_ago'] = (last_date - df['date']).dt.days
    alpha = 0.005 
    df['weight'] = np.exp(-alpha * df['days_ago'])
    
    avg_home = np.average(df['home_goals'], weights=df['weight'])
    avg_away = np.average(df['away_goals'], weights=df['weight'])
    
    team_stats = {}
    all_teams = sorted(list(set(df['home'].unique()) | set(df['away'].unique())))
    
    for team in all_teams:
        h_m = df[df['home'] == team]
        if not h_m.empty:
            att_h = np.average(h_m['home_goals'], weights=h_m['weight']) / avg_home
            def_h = np.average(h_m['away_goals'], weights=h_m['weight']) / avg_away
        else: att_h, def_h = 1.0, 1.0

        a_m = df[df['away'] == team]
        if not a_m.empty:
            att_a = np.average(a_m['away_goals'], weights=a_m['weight']) / avg_away
            def_a = np.average(a_m['home_goals'], weights=a_m['weight']) / avg_home
        else: att_a, def_a = 1.0, 1.0
            
        team_stats[team] = {'att_h': att_h, 'def_h': def_h, 'att_a': att_a, 'def_a': def_a}
        
    return team_stats, avg_home, avg_away, all_teams

def predict_match_dixon_coles(home, away, team_stats, avg_h, avg_a):
    h_exp = team_stats[home]['att_h'] * team_stats[away]['def_a'] * avg_h
    a_exp = team_stats[away]['att_a'] * team_stats[home]['def_h'] * avg_a
    
    max_goals = 10
    probs = np.zeros((max_goals, max_goals))
    rho = -0.13 

    for x in range(max_goals):
        for y in range(max_goals):
            p_base = poisson.pmf(x, h_exp) * poisson.pmf(y, a_exp)
            correction = 1.0
            if x==0 and y==0: correction = 1.0 - (h_exp * a_exp * rho)
            elif x==0 and y==1: correction = 1.0 + (h_exp * rho)
            elif x==1 and y==0: correction = 1.0 + (a_exp * rho)
            elif x==1 and y==1: correction = 1.0 - (rho)
            probs[x][y] = p_base * correction
            
    probs = np.maximum(0, probs)
    probs = probs / probs.sum()

    p_home = np.tril(probs, -1).sum()
    p_draw = np.diag(probs).sum()
    p_away = np.triu(probs, 1).sum()
    
    p_o25 = 0
    for i in range(max_goals):
        for j in range(max_goals):
            if (i+j) > 2.5: p_o25 += probs[i][j]

    flat_indices = np.argsort(probs.ravel())[::-1][:3]
    top_scores = []
    for idx in flat_indices:
        i, j = np.unravel_index(idx, probs.shape)
        top_scores.append((f"{i}-{j}", probs[i][j]))

    return h_exp, a_exp, p_home, p_draw, p_away, p_o25, top_scores

# --- B. LÓGICA XGBOOST (IA) ---
def calculate_rolling_stats(df, window=5):
    """Calcula stats de los últimos N partidos"""
    home_df = df[['date', 'home', 'home_goals', 'away_goals']].copy()
    home_df.columns = ['date', 'team', 'scored', 'conceded']
    
    away_df = df[['date', 'away', 'away_goals', 'home_goals']].copy()
    away_df.columns = ['date', 'team', 'scored', 'conceded']
    
    stats_df = pd.concat([home_df, away_df]).sort_values(['team', 'date'])
    
    # Rolling mean con shift para no ver el futuro
    stats_df['avg_scored_l5'] = stats_df.groupby('team')['scored'].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )
    stats_df['avg_conceded_l5'] = stats_df.groupby('team')['conceded'].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )
    stats_df.fillna(stats_df.mean(numeric_only=True), inplace=True)
    return stats_df

def prepare_features_xgboost(df):
    data = df.copy()
    rolling_stats = calculate_rolling_stats(data, window=5)
    
    # Merge Home Stats
    data = data.merge(rolling_stats[['date', 'team', 'avg_scored_l5', 'avg_conceded_l5']], 
                      left_on=['date', 'home'], right_on=['date', 'team'], how='left')
    data.rename(columns={'avg_scored_l5': 'home_att_l5', 'avg_conceded_l5': 'home_def_l5'}, inplace=True)
    data.drop('team', axis=1, inplace=True)
    
    # Merge Away Stats
    data = data.merge(rolling_stats[['date', 'team', 'avg_scored_l5', 'avg_conceded_l5']], 
                      left_on=['date', 'away'], right_on=['date', 'team'], how='left')
    data.rename(columns={'avg_scored_l5': 'away_att_l5', 'avg_conceded_l5': 'away_def_l5'}, inplace=True)
    data.drop('team', axis=1, inplace=True)
    
    # Target (0: Home, 1: Draw, 2: Away)
    conditions = [
        (data['home_goals'] > data['away_goals']),
        (data['home_goals'] == data['away_goals']),
        (data['home_goals'] < data['away_goals'])
    ]
    data['target'] = np.select(conditions, [0, 1, 2])
    
    # Encoding
    le = LabelEncoder()
    all_teams = list(set(data['home']) | set(data['away']))
    le.fit(all_teams)
    data['home_code'] = le.transform(data['home'])
    data['away_code'] = le.transform(data['away'])
    
    features = ['home_code', 'away_code', 'odd_h', 'odd_d', 'odd_a', 
                'home_att_l5', 'home_def_l5', 'away_att_l5', 'away_def_l5']
    
    return data, features, le

def train_xgboost_model(df):
    data, feature_cols, encoder = prepare_features_xgboost(df)
    X = data[feature_cols]
    y = data['target']
    
    # Split temporal (entrenar con pasado, testear con reciente)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model = xgb.XGBClassifier(
        n_estimators=100, learning_rate=0.05, max_depth=3,
        objective='multi:softprob', num_class=3, eval_metric='mlogloss', random_state=42
    )
    model.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, encoder, feature_cols, acc

def predict_match_xgboost(model, encoder, home_team, away_team, odd_h, odd_d, odd_a, df_historico):
    try:
        h_code = encoder.transform([home_team])[0]
        a_code = encoder.transform([away_team])[0]
    except: return 0,0,0,0,0 # Equipos nuevos desconocidos

    # Calcular rolling stats actuales
    stats_df = calculate_rolling_stats(df_historico, window=5)
    
    try:
        h_stats = stats_df[stats_df['team'] == home_team].iloc[-1]
        a_stats = stats_df[stats_df['team'] == away_team].iloc[-1]
        
        h_att, h_def = h_stats['avg_scored_l5'], h_stats['avg_conceded_l5']
        a_att, a_def = a_stats['avg_scored_l5'], a_stats['avg_conceded_l5']
    except:
        h_att, h_def, a_att, a_def = 1.0, 1.0, 1.0, 1.0 # Default fallback

    input_data = pd.DataFrame([[
        h_code, a_code, odd_h, odd_d, odd_a, h_att, h_def, a_att, a_def
    ]], columns=['home_code', 'away_code', 'odd_h', 'odd_d', 'odd_a', 
                 'home_att_l5', 'home_def_l5', 'away_att_l5', 'away_def_l5'])
    
    probs = model.predict_proba(input_data)[0]
    return probs[0], probs[1], probs[2], h_att, a_att

# ======================================================
# 3. UTILIDADES VISUALES Y GESTIÓN 🛠️
# ======================================================
def plot_gauge(val, title, color):
    return go.Figure(go.Indicator(
        mode="gauge+number", value=val*100, title={'text': title},
        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': color}, 'bgcolor': "white"}
    )).update_layout(height=150, margin=dict(l=20, r=20, t=30, b=20))

def plot_goals_evolution(df, home_team, away_team, n=10):
    def get_team_series(team):
        mask = (df['home'] == team) | (df['away'] == team)
        matches = df[mask].sort_values(by='date').tail(n)
        dates = matches['date'].dt.strftime('%d/%m').tolist()
        goals, rivals = [], []
        for _, row in matches.iterrows():
            if row['home'] == team:
                goals.append(row['home_goals'])
                rivals.append(f"vs {row['away']} (L)")
            else:
                goals.append(row['away_goals'])
                rivals.append(f"vs {row['home']} (V)")
        return dates, goals, rivals

    d1, g1, r1 = get_team_series(home_team)
    d2, g2, r2 = get_team_series(away_team)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d1, y=g1, mode='lines+markers', name=home_team,
                             line=dict(color='#4CAF50', width=3), text=r1, hoverinfo='text+y'))
    fig.add_trace(go.Scatter(x=d2, y=g2, mode='lines+markers', name=away_team,
                             line=dict(color='#2196F3', width=3, dash='dot'), text=r2, hoverinfo='text+y'))
    
    fig.update_layout(title="Evolución Goles (Últimos 10 PJ)", paper_bgcolor='rgba(0,0,0,0)', 
                      plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
                      legend=dict(orientation="h", y=1.1), height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def get_last_5(df, team):
    mask = (df['home'] == team) | (df['away'] == team)
    l5 = df[mask].sort_values(by='date', ascending=False).head(5).copy()
    l5['Rival'] = np.where(l5['home'] == team, l5['away'], l5['home'])
    l5['Score'] = l5['home_goals'].astype(int).astype(str) + "-" + l5['away_goals'].astype(int).astype(str)
    l5['Sede'] = np.where(l5['home'] == team, '🏠', '✈️')
    return l5[['Sede', 'Rival', 'Score']]

def calculate_kelly(prob, odd):
    if prob <= 0 or odd <= 1: return 0.0
    b = odd - 1
    f = (b * prob - (1 - prob)) / b
    return max(0.0, f * 0.5) * 100

def manage_bets(mode, data=None, id_bet=None, status=None):
    if os.path.exists(CSV_FILE): df = pd.read_csv(CSV_FILE)
    else: df = pd.DataFrame(columns=["ID", "Fecha", "Liga", "Partido", "Pick", "Cuota", "Stake", "Prob", "Estado", "Ganancia"])
    
    if mode == "save":
        df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
        df.to_csv(CSV_FILE, index=False)
    elif mode == "update":
        idx = df[df['ID'].astype(str) == str(id_bet)].index
        if not idx.empty:
            i = idx[0]
            df.at[i, 'Estado'] = status
            profit = (df.at[i, 'Stake'] * df.at[i, 'Cuota']) - df.at[i, 'Stake'] if status == 'Ganada' else (-df.at[i, 'Stake'] if status == 'Perdida' else 0)
            df.at[i, 'Ganancia'] = profit
            df.to_csv(CSV_FILE, index=False)
    return df

def run_backtest_dc(df, team_stats, avg_h, avg_a):
    recent = df.tail(50).copy()
    results = []
    correct, bal = 0, 0
    for _, row in recent.iterrows():
        _, _, ph, pd_prob, pa, _, _ = predict_match_dixon_coles(row['home'], row['away'], team_stats, avg_h, avg_a)
        if ph > pd_prob and ph > pa: pred, prob, odd, res_real = "Local", ph, row['odd_h'], ("Local" if row['home_goals'] > row['away_goals'] else "Fallo")
        elif pa > ph and pa > pd_prob: pred, prob, odd, res_real = "Visita", pa, row['odd_a'], ("Visita" if row['away_goals'] > row['home_goals'] else "Fallo")
        else: pred, prob, odd, res_real = "Empate", pd_prob, row['odd_d'], ("Empate" if row['home_goals'] == row['away_goals'] else "Fallo")
        
        is_win = (pred == res_real)
        profit = (odd - 1) if is_win else -1
        if is_win: correct += 1
        bal += profit
        results.append({"Partido": f"{row['home']} vs {row['away']}", "Pred": pred, "Cuota": odd, "Res": "✅" if is_win else "❌"})
    return pd.DataFrame(results), correct, bal

# ======================================================
# 4. INTERFAZ GRÁFICA (UI) 🌟
# ======================================================
with st.sidebar:
    st.header("⚙️ Configuración")
    
    # Selector de Liga
    leagues = {"SP1": "🇪🇸 La Liga", "E0": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier", "I1": "🇮🇹 Serie A", "D1": "🇩🇪 Bundesliga", "F1": "🇫🇷 Ligue 1"}
    code = st.selectbox("Liga", list(leagues.keys()), format_func=lambda x: leagues[x])
    
    # Carga de Datos
    df = fetch_live_soccer_data(code)
    if not df.empty:
        # Calcular stats para Dixon-Coles (siempre útil tenerlo)
        stats, ah, aa, teams = calculate_strengths(df)
        st.success(f"✅ {len(df)} partidos cargados")
    else: st.error("Error cargando datos"); st.stop()
    
    st.divider()
    
    # --- SELECTOR DE MODELO ---
    model_mode = st.radio("🤖 Modelo IA", ["Dixon-Coles (Estadístico)", "XGBoost (Machine Learning)"])
    
    st.divider()
    bank = st.number_input("💰 Tu Banco ($)", 1000.0, step=50.0)
    
    if st.session_state.ticket:
        st.divider()
        st.markdown(f"### 🎫 Ticket ({len(st.session_state.ticket)})")
        if st.button("🗑️ Limpiar Ticket"):
            st.session_state.ticket = []; st.rerun()

st.title(f"⚽ {leagues[code]} - {model_mode}")

# Selección de equipos
c1, c2 = st.columns(2)
home = c1.selectbox("Local", teams)
away = c2.selectbox("Visitante", [t for t in teams if t != home])

# --- EJECUCIÓN DEL MODELO SELECCIONADO ---
if model_mode == "Dixon-Coles (Estadístico)":
    h_exp, a_exp, ph, pd_prob, pa, po25, top_sc = predict_match_dixon_coles(home, away, stats, ah, aa)
    xg_metrics = None
else:
    # XGBoost
    with st.spinner("Entrenando IA con datos históricos..."):
        xgb_model, encoder, cols, acc_score = train_xgboost_model(df)
    
    st.info(f"🧠 Precisión del modelo en Test: **{acc_score*100:.1f}%**")
    
    # Inputs de cuotas actuales (necesarios para XGBoost)
    col_odd_input1, col_odd_input2, col_odd_input3 = st.columns(3)
    curr_oh = col_odd_input1.number_input("Cuota Local (Actual)", 1.01, 20.0, 2.50)
    curr_od = col_odd_input2.number_input("Cuota Empate (Actual)", 1.01, 20.0, 3.20)
    curr_oa = col_odd_input3.number_input("Cuota Visita (Actual)", 1.01, 20.0, 2.90)
    
    ph, pd_prob, pa, h_att_val, a_att_val = predict_match_xgboost(xgb_model, encoder, home, away, curr_oh, curr_od, curr_oa, df)
    
    # Valores dummy para visualización (XGBoost no predice goles exactos directamente en este modo)
    h_exp = 0.0; a_exp = 0.0; po25 = 0.0; top_sc = [("-", 0), ("-", 0)]
    xg_metrics = (h_att_val, a_att_val) # Guardamos para mostrar luego

# PESTAÑAS
t1, t2, t3, t4 = st.tabs(["📊 Análisis", "💰 Valor y Parlay", "📜 Historial", "🧪 Laboratorio"])

with t1:
    if model_mode == "XGBoost (Machine Learning)":
        st.markdown("### 📊 Estado de Forma (Goles anotados - Últimos 5)")
        c_m1, c_m2 = st.columns(2)
        c_m1.metric(f"Ataque {home}", f"{xg_metrics[0]:.2f}", delta="Promedio L5")
        c_m2.metric(f"Ataque {away}", f"{xg_metrics[1]:.2f}", delta="Promedio L5")
    else:
        st.markdown("### 🥅 Expectativa de Goles (Poisson)")
        c_g1, c_g2, c_g3 = st.columns(3)
        c_g1.metric(home, f"{h_exp:.2f}")
        c_g2.metric("Total", f"{h_exp+a_exp:.2f}", delta=f"Over 2.5: {po25*100:.0f}%")
        c_g3.metric(away, f"{a_exp:.2f}")

    st.markdown("### 🏆 Probabilidades de Victoria")
    g1, g2, g3 = st.columns(3)
    g1.plotly_chart(plot_gauge(ph, f"Gana {home}", "#4CAF50"), use_container_width=True)
    g2.plotly_chart(plot_gauge(pd_prob, "Empate", "#FFC107"), use_container_width=True)
    g3.plotly_chart(plot_gauge(pa, f"Gana {away}", "#2196F3"), use_container_width=True)

    st.divider()
    st.markdown("### 📈 Tendencia de Goles")
    fig_evol = plot_goals_evolution(df, home, away, n=10)
    st.plotly_chart(fig_evol, use_container_width=True)
    
    st.markdown("### 📉 Últimos Resultados")
    cf1, cf2 = st.columns(2)
    with cf1: st.dataframe(get_last_5(df, home), use_container_width=True, hide_index=True)
    with cf2: st.dataframe(get_last_5(df, away), use_container_width=True, hide_index=True)

with t2:
    col_analisis, col_ticket = st.columns([2, 1])
    with col_analisis:
        st.markdown("### 🏦 Calculadora de Valor")
        # Cuotas para calcular EV
        if model_mode == "Dixon-Coles (Estadístico)":
            co1, co2, co3 = st.columns(3)
            oh = co1.number_input("Cuota 1", 1.01, 20.0, 2.0, key="dc_o1")
            od = co2.number_input("Cuota X", 1.01, 20.0, 3.2, key="dc_ox")
            oa = co3.number_input("Cuota 2", 1.01, 20.0, 3.5, key="dc_o2")
        else:
            oh, od, oa = curr_oh, curr_od, curr_oa # Ya ingresadas arriba para XGBoost

        # Kelly y EV
        ev_h, kh = (ph*oh)-1, calculate_kelly(ph, oh)
        ev_d, kd = (pd_prob*od)-1, calculate_kelly(pd_prob, od)
        ev_a, ka = (pa*oa)-1, calculate_kelly(pa, oa)

        def card(lab, ev, k, odd):
            color = "green" if ev > 0 else "red"
            st.markdown(f"""
            <div style="border:1px solid {color}; padding:10px; border-radius:5px; margin-bottom:5px;">
                <strong>{lab}</strong> (@{odd})<br>
                EV: {ev*100:.1f}% | Kelly: {k:.1f}%
            </div>
            """, unsafe_allow_html=True)
            
        cv1, cv2, cv3 = st.columns(3)
        with cv1: card(home, ev_h, kh, oh)
        with cv2: card("Empate", ev_d, kd, od)
        with cv3: card(away, ev_a, ka, oa)

        st.divider()
        with st.form("add_to_ticket"):
            sel_pick = st.selectbox("Selección", [f"Gana {home}", "Empate", f"Gana {away}"])
            if "Gana "+home in sel_pick: sel_odd, sel_prob = oh, ph
            elif "Empate" in sel_pick: sel_odd, sel_prob = od, pd_prob
            else: sel_odd, sel_prob = oa, pa
            
            if st.form_submit_button("➕ Añadir al Ticket"):
                item = {"match": f"{home} vs {away}", "pick": sel_pick, "odd": sel_odd, "prob": sel_prob, "league": leagues[code]}
                st.session_state.ticket.append(item)
                st.rerun()

    with col_ticket:
        st.markdown("### 🎫 Ticket")
        if not st.session_state.ticket:
            st.info("Vacío")
        else:
            total_odd = 1.0
            total_prob = 1.0
            for idx, item in enumerate(st.session_state.ticket):
                c_info, c_del = st.columns([5, 1])
                with c_info:
                    st.markdown(f"""
                    <div class="ticket-box">
                        <small>{item['league']}</small><br><strong>{item['match']}</strong><br>
                        {item['pick']} <span style="color:#4CAF50">@{item['odd']}</span>
                    </div>
                    """, unsafe_allow_html=True)
                with c_del:
                    if st.button("❌", key=f"del_{idx}"):
                        st.session_state.ticket.pop(idx); st.rerun()
                total_odd *= item['odd']
                total_prob *= item['prob']
            
            st.divider()
            st.metric("Cuota Total", f"{total_odd:.2f}")
            st.metric("Prob. Real", f"{total_prob*100:.1f}%")
            stake_parlay = st.number_input("Stake", 1.0, 5000.0, 50.0)
            if st.button("💾 Guardar Apuesta"):
                type_str = "Simple" if len(st.session_state.ticket) == 1 else "Parlay"
                manage_bets("save", {
                    "ID": pd.Timestamp.now().strftime('%Y%m%d%H%M%S'),
                    "Fecha": pd.Timestamp.now().strftime('%Y-%m-%d'), 
                    "Liga": type_str,
                    "Partido": "Combinada" if len(st.session_state.ticket)>1 else st.session_state.ticket[0]['match'],
                    "Pick": " + ".join([i['pick'] for i in st.session_state.ticket]),
                    "Cuota": round(total_odd, 2), "Stake": stake_parlay, "Prob": round(total_prob, 4),
                    "Estado": "Pendiente", "Ganancia": 0.0
                })
                st.session_state.ticket = []; st.success("Guardado!"); st.rerun()

with t3:
    st.markdown("### 📜 Historial")
    db = manage_bets("load")
    if not db.empty:
        st.metric("Balance Total", f"${db['Ganancia'].sum():.2f}")
        st.dataframe(db.sort_values(by="Fecha", ascending=False), use_container_width=True)
        with st.expander("Actualizar Resultados"):
            pen = db[db['Estado']=='Pendiente']
            if not pen.empty:
                bid = st.selectbox("ID", pen['ID'].unique())
                res = st.selectbox("Resultado", ["Ganada", "Perdida", "Push"])
                if st.button("Actualizar"): manage_bets("update", id_bet=bid, status=res); st.rerun()
            else: st.info("No hay pendientes")

with t4:
    st.markdown("### 🧪 Laboratorio")
    st.info("Backtesting de los últimos 50 partidos usando el Modelo Dixon-Coles.")
    if st.button("▶️ Ejecutar Simulación"):
        test_df, ok, profit = run_backtest_dc(df, stats, ah, aa)
        m1, m2, m3 = st.columns(3)
        m1.metric("Aciertos", f"{ok}/50 ({ok/50*100:.0f}%)")
        m2.metric("Profit (Stake 1U)", f"{profit:.2f} U", delta_color="normal")
        st.dataframe(test_df, use_container_width=True)
