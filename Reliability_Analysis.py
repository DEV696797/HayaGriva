import streamlit as st
import pandas as pd
import numpy as np

st.title("Cronbach Alpha Reliability")

df = st.session_state.get("data")

if df is None:
    st.warning("Upload dataset first")

else:

    emotion_cols = df.columns[:5]

    def cronbach_alpha(data):

        corr = data.corr()

        N = data.shape[1]

        mean_corr = corr.values[np.triu_indices(N,1)].mean()

        alpha = (N * mean_corr) / (1 + (N - 1) * mean_corr)

        return alpha

    alpha = cronbach_alpha(df[emotion_cols])

    st.metric("Cronbach Alpha", round(alpha,3))

    if alpha > 0.8:
        st.success("High reliability confirmed")