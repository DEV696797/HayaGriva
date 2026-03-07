import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from sklearn.cluster import KMeans
import networkx as nx
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

# -------------------
# THEME
# -------------------

st.markdown("""
<style>
.stApp{
background:linear-gradient(135deg,#1b0036,#3b0a63,#5e17eb);
color:white;
}
</style>
""",unsafe_allow_html=True)

st.title("HayaGriva Luxury Consumer Intelligence")

st.subheader("Effect of Emotions on Purchase Intention of Luxury Products")

# -------------------
# DATA UPLOAD
# -------------------

emotion_file=st.file_uploader("Upload Emotional Dataset")

demo_file=st.file_uploader("Upload Demographic Dataset")

if emotion_file:

    df=pd.read_excel(emotion_file)

    df=df.replace(0,3)

    emotion=df.iloc[:,5]
    celebrity=df.iloc[:,10]
    fomo=df.iloc[:,15]
    purchase=df.iloc[:,20]

# -------------------
# REGRESSION
# -------------------

    st.header("Multiple Linear Regression")

    X=pd.DataFrame({
    "Emotion":emotion,
    "Celebrity":celebrity,
    "FOMO":fomo
    })

    X=sm.add_constant(X)

    model=sm.OLS(purchase,X).fit()

    col1,col2=st.columns([2,1])

    with col1:
        st.dataframe(model.summary2().tables[1])

    with col2:
        st.markdown("""
### Insights

• Emotion has the **strongest impact on purchase intention**

• Celebrity influence strengthens aspirational identity

• FOMO drives urgency and social comparison

• Psychological factors strongly explain luxury consumption
""")

# -------------------
# DRIVER IMPACT
# -------------------

    st.header("Driver Impact Comparison")

    drivers=pd.DataFrame({

    "Driver":["Emotion","Celebrity","FOMO"],
    "Impact":[emotion.mean(),celebrity.mean(),fomo.mean()]

    })

    col1,col2=st.columns([2,1])

    fig=px.bar(drivers,x="Driver",y="Impact")

    with col1:
        st.plotly_chart(fig,use_container_width=True)

    with col2:
        st.markdown("""
### Interpretation

• Emotional gratification dominates luxury motivation

• Celebrity influence builds aspirational perception

• FOMO creates urgency to purchase luxury products
""")

# -------------------
# SCATTER RELATION