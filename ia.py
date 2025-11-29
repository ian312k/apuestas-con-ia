import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import plotly.graph_objects as go
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ======================================================
# 1. CONFIGURACIÓN Y ESTILOS
# ======================================================
st.set_page_config(page_title="AI Betting Suite", layout="wide", page_icon="🚀")
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
# 2. INGENIERÍA DE DATOS Y UTILIDADES
# ======================================================
@st.cache_data
def fetch_live_soccer_data(league_code="SP1"):
    url = f"https://www.football-data.co.uk/mmz4281/2526/{league_code}.csv"
    try:
        df = pd.read_csv(url)
        cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'B365H', 'B365D', 'B365A', 'HST', 'AST']
        actual = [c for c in cols if c in df.columns]
        df = df[actual]
        mapping = {'Date': 'date', 'HomeTeam': 'home', 'AwayTeam': 'away', 'FTHG': 'home_goals', 'FTAG': 'away_goals', 
                   'B365H': 'odd_h', 'B365D': 'odd_d', 'B365A': 'odd_a', 'HST': 'home_shots', 'AST': 'away_shots'}
        df = df.rename(columns=mapping).dropna().sort_values('date')
        df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
        # Target: 1=Home, 2=Away, 0=Draw
        conditions = [(df['home_goals'] > df['away_goals']), (df['home_goals'] < df['away_goals'])]
        df['result'] = np.select(conditions, [1, 2], default=0)
        return df
    except: return pd.DataFrame()

def calculate_rolling_features(df):
    """Crea estadísticas de forma reciente (últimos 3 partidos)"""
    # Preparar DF vertical
    h_df = df[['date', 'home', 'home_goals', 'away_goals', 'result']].rename(columns={'home':'team', 'home_goals':'gf', 'away_goals':'ga', 'result':'res'})
    h_df['pts'] = np.where(h_df['res']==1, 3, np.where(h_df['res']==0, 1, 0))
    a_df = df[['date', 'away', 'away_goals', 'home_goals', 'result']].rename(columns={'away':'team', 'away_goals':'gf', 'home_goals':'ga', 'result':'res'})
    a_df['pts'] = np.where(a_df['res']==2, 3, np.where(a_df['res']==0, 1, 0))
    
    stats = pd.concat([h_df, a_df]).sort_values(['team', 'date'])
    
    # Rolling 3
    for col in ['pts', 'gf', 'ga']:
        stats[f'roll_{col}'] = stats.groupby('team')[col].transform(lambda x: x.rolling(3, min_periods=1).mean().shift(1)).fillna(0)
    
    # Merge back
    df = df.merge(stats[['date', 'team', 'roll_pts', 'roll_gf', 'roll_ga']], left_on=['date', 'home'], right_on=['date', 'team'], how='left').rename(columns={'roll_pts':'h_form', 'roll_gf':'h_att', 'roll_ga':'h_def'}).drop(columns=['team'])
    df = df.merge(stats[['date', 'team', 'roll_pts', 'roll_gf', 'roll_ga']], left_on=['date', 'away'], right_on=['date', 'team'], how='left').rename(columns={'roll_pts':'a_form', 'roll_gf':'a_att', 'roll_ga':'a_def'}).drop(columns=['team'])
    return df, stats

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

def calculate_kelly(prob, odd):
    if prob <= 0 or odd <= 1: return 0.0
    b = odd - 1
    f = (b * prob - (1 - prob)) / b
    return max(0.0, f * 0.5) * 100

def get_last_5(df, team):
    mask = (df['home'] == team) | (df['away'] == team)
    l5 = df[mask].sort_values(by='date', ascending=False).head(5).copy()
    l5['Rival'] = np.where(l5['home'] == team, l5['away'], l5['home'])
    l5['Score'] = l5['home_goals'].astype(int).astype(str) + "-" + l5['away_goals'].astype(int).astype(str)
    l5['Sede'] = np.where(l5['home'] == team, '🏠', '✈️')
    return l5[['Sede', 'Rival', 'Score']]

def plot_gauge(val, title, color):
    return go.Figure(go.Indicator(
        mode="gauge+number", value=val*100, title={'text': title},
        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': color}, 'bgcolor': "white"}
    )).update_layout(height=150, margin=dict(l=20, r=20, t=30, b=20))

# ======================================================
# 3. MODELOS (AI & DIXON-COLES)
# ======================================================
# --- Dixon Coles ---
def calculate_dc_stats(df):
    last = df['date'].max()
    df['days'] = (last - df['date']).dt.days
    df['w'] = np.exp(-0.005 * df['days'])
    avg_h = np.average(df['home_goals'], weights=df['w'])
    avg_a = np.average(df['away_goals'], weights=df['w'])
    stats = {}
    for t in set(df['home']) | set(df['away']):
        hm = df[df['home']==t]
        am = df[df['away']==t]
        att_h = np.average(hm['home_goals'], weights=hm['w'])/avg_h if not hm.empty else 1.0
        def_h = np.average(hm['away_goals'], weights=hm['w'])/avg_a if not hm.empty else 1.0
        att_a = np.average(am['away_goals'], weights=am['w'])/avg_a if not am.empty else 1.0
        def_a = np.average(am['home_goals'], weights=am['w'])/avg_h if not am.empty else 1.0
        stats[t] = {'att_h': att_h, 'def_h': def_h, 'att_a': att_a, 'def_a': def_a}
    return stats, avg_h, avg_a

def predict_dc(home, away, stats, avg_h, avg_a):
    he = stats[home]['att_h'] * stats[away]['def_a'] * avg_h
    ae = stats[away]['att_a'] * stats[home]['def_h'] * avg_a
    probs = np.zeros((10,10))
    rho = -0.13
    for x in range(10):
        for y in range(10):
            p = poisson.pmf(x, he) * poisson.pmf(y, ae)
            c = 1.0
            if x==0 and y==0: c = 1-(he*ae*rho)
            elif x==0 and y==1: c = 1+(he*rho)
            elif x==1 and y==0: c = 1+(ae*rho)
            elif x==1 and y==1: c = 1-rho
            probs[x][y] = p*c
    probs = np.maximum(0, probs)
    probs /= probs.sum()
    return np.tril(probs,-1).sum(), np.diag(probs).sum(), np.triu(probs,1).sum(), he, ae

# --- Random Forest ---
def train_rf(df):
    le = LabelEncoder()
    le.fit(pd.concat([df['home'], df['away']]).unique())
    df['hc'] = le.transform(df['home'])
    df['ac'] = le.transform(df['away'])
    feats = ['hc', 'ac', 'odd_h', 'odd_d', 'odd_a', 'h_form', 'h_att', 'h_def', 'a_form', 'a_att', 'a_def']
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(df[feats], df['result'])
    return model, le

def predict_rf(model, le, home, away, oh, od, oa, stats_df):
    try:
        hc, ac = le.transform([home])[0], le.transform([away])[0]
        # Get recent stats
        h_st = stats_df[stats_df['team']==home].tail(1)
        a_st = stats_df[stats_df['team']==away].tail(1)
        
        # Safe fetch
        hf, ha, hd = (h_st.iloc[0]['roll_pts']*3, h_st.iloc[0]['roll_gf'], h_st.iloc[0]['roll_ga']) if not h_st.empty else (0,0,0)
        af, aa, ad = (a_st.iloc[0]['roll_pts']*3, a_st.iloc[0]['roll_gf'], a_st.iloc[0]['roll_ga']) if not a_st.empty else (0,0,0)

        in_data = pd.DataFrame([[hc, ac, oh, od, oa, hf, ha, hd, af, aa, ad]], 
                               columns=['hc', 'ac', 'odd_h', 'odd_d', 'odd_a', 'h_form', 'h_att', 'h_def', 'a_form', 'a_att', 'a_def'])
        probs = model.predict_proba(in_data)[0]
        
        # Map classes (0=Draw, 1=Home, 2=Away)
        pd_p, ph_p, pa_p = 0.0, 0.0, 0.0
        for i, c in enumerate(model.classes_):
            if c==0: pd_p=probs[i]
            if c==1: ph_p=probs[i]
            if c==2: pa_p=probs[i]
        return ph_p, pd_p, pa_p
    except: return 0.33, 0.33, 0.34

def run_backtest(df, model_type, model_obj=None, le=None, dc_stats=None):
    recent = df.tail(20).copy()
    log = []
    bal = 0
    correct = 0
    
    for _, r in recent.iterrows():
        # Predict based on selected model
        if "IA" in model_type:
            # Note: Testing on training data is optimistic, but valid for checking model logic
            ph, pd_p, pa = predict_rf(model_obj, le, r['home'], r['away'], r['odd_h'], r['odd_d'], r['odd_a'], df) # using df as stats source for simplicity
        else:
            ph, pd_p, pa, _, _ = predict_dc(r['home'], r['away'], dc_stats[0], dc_stats[1], dc_stats[2])
        
        # Decide pick
        if ph > pd_p and ph > pa: pick, prob, odd, res_txt = "Local", ph, r['odd_h'], ("Local" if r['result']==1 else "Fallo")
        elif pa > ph and pa > pd_p: pick, prob, odd, res_txt = "Visita", pa, r['odd_a'], ("Visita" if r['result']==2 else "Fallo")
        else: pick, prob, odd, res_txt = "Empate", pd_p, r['odd_d'], ("Empate" if r['result']==0 else "Fallo")
        
        is_win = (pick == res_txt)
        profit = (odd - 1) if is_win else -1
        bal += profit
        if is_win: correct += 1
        
        log.append({"Partido": f"{r['home']} vs {r['away']}", "Pred": f"{pick} ({prob*100:.0f}%)", "Real": res_txt, "Cuota": odd, "P/L": round(profit, 2)})
        
    return pd.DataFrame(log), correct, bal

# ======================================================
# 4. INTERFAZ GRÁFICA (RESTORED TABS)
# ======================================================
with st.sidebar:
    st.header("⚙️ Configuración")
    leagues = {"SP1": "🇪🇸 La Liga", "E0": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League", "I1": "🇮🇹 Serie A", "D1": "🇩🇪 Bundesliga"}
    code = st.selectbox("Liga", list(leagues.keys()), format_func=lambda x: leagues[x])
    
    raw = fetch_live_soccer_data(code)
    if raw.empty: st.error("Error datos"); st.stop()
    
    # Process
    df_pro, stats_hist = calculate_rolling_features(raw)
    
    st.divider()
    m_type = st.radio("Cerebro:", ["Dixon-Coles (Estadístico)", "Random Forest (IA)"], index=1)
    
    # Train/Prep Models
    if "IA" in m_type:
        rf_model, encoder = train_rf(df_pro)
        st.success(f"🚀 IA Entrenada ({len(df_pro)} partidos)")
    else:
        dc_s, avg_h, avg_a = calculate_dc_stats(raw)
        st.success("📐 Dixon-Coles Listo")
    
    st.divider()
    bank = st.number_input("💰 Tu Banco ($)", 1000.0, step=50.0)

st.title(f"⚽ {leagues[code]} - {m_type}")

# Selectors
teams = sorted(raw['home'].unique())
c1, c2 = st.columns(2)
h_tm = c1.selectbox("Local", teams, index=0)
a_tm = c2.selectbox("Visitante", [t for t in teams if t != h_tm], index=0)

# Odds Input
st.info("ℹ️ Ingresa cuotas reales para activar la predicción")
co1, co2, co3 = st.columns(3)
oh = co1.number_input("Cuota 1", 1.01, 20.0, 2.0)
od = co2.number_input("Cuota X", 1.01, 20.0, 3.2)
oa = co3.number_input("Cuota 2", 1.01, 20.0, 3.5)

# --- PREDICTION ---
he, ae = 0, 0
if "IA" in m_type:
    ph, pd_p, pa = predict_rf(rf_model, encoder, h_tm, a_tm, oh, od, oa, stats_hist)
    msg = "IA analizando Cuotas + Forma Reciente (Pts/Goles)"
else:
    ph, pd_p, pa, he, ae = predict_dc(h_tm, a_tm, dc_s, avg_h, avg_a)
    msg = "Modelo matemático puro (Goles Esperados)"

# --- TABS (THEY ARE BACK!) ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Análisis", "💰 Valor & Apuesta", "📜 Historial", "🧪 Laboratorio"])

with tab1:
    st.markdown("### 🏆 Probabilidades de Partido")
    c1, c2, c3 = st.columns(3)
    c1.plotly_chart(plot_gauge(ph, f"Gana {h_tm}", "#4CAF50"), use_container_width=True)
    c2.plotly_chart(plot_gauge(pd_p, "Empate", "#FFC107"), use_container_width=True)
    c3.plotly_chart(plot_gauge(pa, f"Gana {a_tm}", "#2196F3"), use_container_width=True)
    
    st.caption(f"🧠 Lógica: {msg}")
    
    if he > 0: # Only for Dixon Coles
        st.markdown("### 🥅 Goles Esperados")
        m1, m2, m3 = st.columns(3)
        m1.metric(h_tm, f"{he:.2f}")
        m2.metric("Total", f"{he+ae:.2f}")
        m3.metric(a_tm, f"{ae:.2f}")

    st.markdown("### 📉 Forma Reciente (Últimos 5)")
    cf1, cf2 = st.columns(2)
    with cf1: st.dataframe(get_last_5(raw, h_tm), use_container_width=True, hide_index=True)
    with cf2: st.dataframe(get_last_5(raw, a_tm), use_container_width=True, hide_index=True)

with tab2:
    st.markdown("### 🏦 Cazador de Valor")
    # Kelly Calculation
    ev_h, kh = (ph*oh)-1, calculate_kelly(ph, oh)
    ev_d, kd = (pd_p*od)-1, calculate_kelly(pd_p, od)
    ev_a, ka = (pa*oa)-1, calculate_kelly(pa, oa)
    
    def card(lab, ev, k, odd):
        if ev > 0:
            st.success(f"✅ **{lab}** (+{ev*100:.1f}%)")
            st.markdown(f"**Apostar:** ${bank*(k/100):.2f} ({k:.1f}%)")
        else: st.error(f"❌ **{lab}** (EV {ev*100:.1f}%)")
    
    cv1, cv2, cv3 = st.columns(3)
    with cv1: card(f"{h_tm}", ev_h, kh, oh)
    with cv2: card("Empate", ev_d, kd, od)
    with cv3: card(f"{a_tm}", ev_a, ka, oa)
    
    st.divider()
    st.subheader("📝 Registrar Apuesta")
    with st.form("bet"):
        pk = st.selectbox("Pick", [f"Gana {h_tm}", "Empate", f"Gana {a_tm}"])
        stk = st.number_input("Stake $", 1.0, 5000.0, 50.0)
        if "Gana "+h_tm in pk: fo, fp = oh, ph
        elif "Empate" in pk: fo, fp = od, pd_p
        else: fo, fp = oa, pa
        
        if st.form_submit_button("💾 Guardar"):
            manage_bets("save", {"ID": pd.Timestamp.now().strftime('%Y%m%d%H%M%S'), "Fecha": pd.Timestamp.now().strftime('%Y-%m-%d'), 
                                 "Liga": code, "Partido": f"{h_tm}-{a_tm}", "Pick": pk, "Cuota": fo, "Stake": stk, "Prob": round(fp, 4), "Estado": "Pendiente", "Ganancia": 0.0})
            st.success("Guardado!"); st.rerun()

with tab3:
    st.markdown("### 📜 Historial de Apuestas")
    db = manage_bets("load")
    if not db.empty:
        st.metric("Balance Total", f"${db['Ganancia'].sum():.2f}", delta_color="normal")
        st.dataframe(db.sort_values(by="Fecha", ascending=False), use_container_width=True)
        with st.expander("Actualizar Estado"):
            pen = db[db['Estado']=='Pendiente']
            if not pen.empty:
                bid = st.selectbox("ID", pen['ID'].unique())
                res = st.selectbox("Resultado", ["Ganada", "Perdida", "Push"])
                if st.button("Actualizar"): manage_bets("update", id_bet=bid, status=res); st.rerun()
            else: st.info("No hay pendientes")
    else: st.info("Historial vacío")

with tab4:
    st.markdown("### 🧪 Laboratorio de Backtesting")
    st.info("Pon a prueba el modelo seleccionado con los últimos 20 partidos.")
    if st.button("▶️ Ejecutar Simulación"):
        # Prepare inputs based on model type
        obj = rf_model if "IA" in m_type else None
        enc = encoder if "IA" in m_type else None
        dcs = (dc_s, avg_h, avg_a) if "Dixon" in m_type else None
        
        test_df, ok, profit = run_backtest(df_pro, m_type, obj, enc, dcs)
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Aciertos", f"{ok}/20 ({ok/20*100:.0f}%)")
        m2.metric("Profit (Stake 1U)", f"{profit:.2f} U", delta="Ganancia" if profit>0 else "Pérdida")
        m3.metric("Estado", "🔥 Rentable" if profit > 0 else "❄️ Frío")
        st.dataframe(test_df, use_container_width=True)
