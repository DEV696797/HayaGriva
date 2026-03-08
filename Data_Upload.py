import streamlit as st
import pandas as pd

st.title("Upload Dataset")

file = st.file_uploader("Upload Excel Dataset", type=["xlsx"])

if file:

    df = pd.read_excel(file)

    df = df.apply(pd.to_numeric, errors="coerce")

    df = df.dropna()

    st.session_state["data"] = df

    st.subheader("Dataset Preview")

    st.dataframe(df.head())