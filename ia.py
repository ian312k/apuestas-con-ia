with t1:
    st.markdown("### 🥅 Expectativa de Goles")
    c_g1, c_g2, c_g3 = st.columns(3)
    c_g1.metric(home, f"{h_exp:.2f}")
    # Eliminamos el delta="Over 2.5..." de aquí para limpiarlo
    c_g2.metric("Total (xG)", f"{h_exp+a_exp:.2f}") 
    c_g3.metric(away, f"{a_exp:.2f}")

    # --- SECCIÓN: MERCADOS DE GOLES (MODIFICADA) ---
    st.markdown("### 📊 Probabilidades de Gol")
    # Cambiamos de 2 columnas a 3 para incluir el Over 2.5 aquí
    mg1, mg2, mg3 = st.columns(3)
    
    mg1.metric("Over 1.5 Goles", f"{po15*100:.1f}%", help="Probabilidad de que haya 2 o más goles en total")
    mg2.metric("Over 2.5 Goles", f"{po25*100:.1f}%", help="Probabilidad de que haya 3 o más goles en total")
    mg3.metric("Ambos Anotan (BTTS)", f"{pbtts*100:.1f}%", help="Probabilidad de que ambos equipos marquen")
    # -----------------------------------------------
    
    st.markdown("### 🏆 Probabilidades 1X2")
    g1, g2, g3 = st.columns(3)
    g1.plotly_chart(plot_gauge(ph, f"Gana {home}", "#4CAF50"), use_container_width=True)
    g2.plotly_chart(plot_gauge(pd_prob, "Empate", "#FFC107"), use_container_width=True)
    g3.plotly_chart(plot_gauge(pa, f"Gana {away}", "#2196F3"), use_container_width=True)
    
    st.info(f"🎯 **Marcador Exacto:** {top_sc[0][0]} ({top_sc[0][1]*100:.1f}%) | **Opción 2:** {top_sc[1][0]}")
    
    st.markdown("### 📉 Estado de Forma")
    cf1, cf2 = st.columns(2)
    with cf1: st.dataframe(get_last_5(df, home), use_container_width=True, hide_index=True)
    with cf2: st.dataframe(get_last_5(df, away), use_container_width=True, hide_index=True)
