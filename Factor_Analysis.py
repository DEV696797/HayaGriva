import streamlit as st
from factor_analyzer import FactorAnalyzer, calculate_kmo, calculate_bartlett_sphericity
import pandas as pd

st.title("Exploratory Factor Analysis")

df = st.session_state.get("data")

if df is None:
    st.warning("Upload dataset first")

else:

    X = df.iloc[:,:5]

    kmo_all, kmo_model = calculate_kmo(X)

    chi_square_value, p_value = calculate_bartlett_sphericity(X)

    st.metric("KMO", round(kmo_model,3))

    st.metric("Bartlett p-value", round(p_value,5))

    fa = FactorAnalyzer()

    fa.fit(X)

    ev, v = fa.get_eigenvalues()

    st.line_chart(ev)