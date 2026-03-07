import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
import networkx as nx
import plotly.graph_objects as go
import numpy as np

st.title("HayaGriva Luxury Consumer Analytics")

uploaded_file = st.file_uploader("Upload Dataset", type=["xlsx","csv"])

if uploaded_file:

    df = pd.read_excel(uploaded_file)

    emotion = df.iloc[:,5]
    celebrity = df.iloc[:,10]
    fomo = df.iloc[:,15]
    purchase = df.iloc[:,20]

    st.subheader("Dataset Preview")
    st.write(df.head())

    def regression(x,y):
        x = np.array(x)
        y = np.array(y)
        b = np.polyfit(x,y,1)
        return b[0]

    st.subheader("Causal Path Model")

    path_CF = regression(celebrity,fomo)
    path_FE = regression(fomo,emotion)
    path_EP = regression(emotion,purchase)
    path_CE = regression(celebrity,emotion)

    st.write("Celebrity → FOMO:",round(path_CF,3))
    st.write("FOMO → Emotion:",round(path_FE,3))
    st.write("Emotion → Purchase:",round(path_EP,3))
    st.write("Celebrity → Emotion:",round(path_CE,3))

    st.subheader("Purchase Simulator")

    emotion_input = st.slider("Emotion",1,20,10)
    celebrity_input = st.slider("Celebrity",1,20,10)
    fomo_input = st.slider("FOMO",1,20,10)

    score = 0.63*emotion_input + 0.16*celebrity_input + 0.26*fomo_input

    st.success(f"Predicted Purchase Score: {round(score,2)}")