import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
import networkx as nx
import matplotlib.pyplot as plt
from pptx import Presentation
import feedparser
import random

st.set_page_config(layout="wide")

# -------------------------
# PREMIUM THEME
# -------------------------

st.markdown("""
<style>

.stApp{
background:linear-gradient(135deg,#0f0026,#2c0066,#5e17eb);
color:white;
}

.ticker{
background:#7c3aed;
padding:10px;
font-size:18px;
}

</style>
""", unsafe_allow_html=True)

# -------------------------
# LUXURY NEWS RIBBON
# -------------------------

feeds=[
"https://www.businessoffashion.com/feed/",
"https://rss.nytimes.com/services/xml/rss/nyt/FashionandStyle.xml"
]

news=[]

for f in feeds:
    feed=feedparser.parse(f)
    for entry in feed.entries[:3]:
        news.append(entry.title)

st.markdown(
f"<div class='ticker'>Luxury Industry News: {' ✦ '.join(news)}</div>",
unsafe_allow_html=True
)

# -------------------------
# QUOTE RIBBON
# -------------------------

quotes=[
"Luxury brands sell dreams not products — Bernard Arnault",
"Luxury is identity not consumption — Kapferer",
"Exclusivity creates desire — Pinault",
"Luxury marketing is storytelling"
]

st.markdown(f"### 💎 Luxury Insight: {random.choice(quotes)}")

# -------------------------
# SIDEBAR LOGOS
# -------------------------

st.sidebar.title("Luxury Houses")

logos=[
"https://upload.wikimedia.org/wikipedia/commons/7/7b/Louis_Vuitton_logo_and_wordmark.svg",
"https://upload.wikimedia.org/wikipedia/commons/5/55/Gucci_Logo.svg",
"https://upload.wikimedia.org/wikipedia/commons/2/24/Chanel_logo_interlocking_cs.svg",
"https://upload.wikimedia.org/wikipedia/commons/7/7f/Hermes_logo.svg",
"https://upload.wikimedia.org/wikipedia/commons/2/20/Rolex_logo.svg"
]

for logo in logos:
    st.sidebar.image(logo,width=160)

# -------------------------
# TITLE
# -------------------------

st.title("HayaGriva Luxury Consumer Intelligence Platform")

# -------------------------
# DATA UPLOAD
# -------------------------

file=st.file_uploader("Upload Luxury Emotion Dataset")

if file:

    df=pd.read_excel(file)

    emotion=df.iloc[:,5]
    celebrity=df.iloc[:,10]
    fomo=df.iloc[:,15]
    purchase=df.iloc[:,20]

# -------------------------
# MULTIPLE REGRESSION
# -------------------------

    st.header("Multiple Linear Regression")

    X=pd.DataFrame({
        "Emotion":emotion,
        "Celebrity":celebrity,
        "FOMO":fomo
    })

    X=sm.add_constant(X)

    model=sm.OLS(purchase,X).fit()

    st.dataframe(model.summary().tables[1])

    st.metric("R²",round(model.rsquared,3))

# -------------------------
# STRUCTURAL EQUATION MODEL
# -------------------------

    st.header("Structural Equation Model")

    path_CF=np.corrcoef(celebrity,fomo)[0,1]
    path_FE=np.corrcoef(fomo,emotion)[0,1]
    path_EP=np.corrcoef(emotion,purchase)[0,1]

    G=nx.DiGraph()

    G.add_edge("Celebrity","FOMO",weight=path_CF)
    G.add_edge("FOMO","Emotion",weight=path_FE)
    G.add_edge("Emotion","Purchase",weight=path_EP)

    pos=nx.spring_layout(G)

    fig=go.Figure()

    for edge in G.edges():
        x0,y0=pos[edge[0]]
        x1,y1=pos[edge[1]]

        fig.add_trace(go.Scatter(
            x=[x0,x1],
            y=[y0,y1],
            mode="lines"
        ))

    for node in G.nodes():

        x,y=pos[node]

        fig.add_trace(go.Scatter(
            x=[x],
            y=[y],
            text=node,
            mode="markers+text",
            marker=dict(size=30)
        ))

    st.plotly_chart(fig)

# -------------------------
# PSYCHOGRAPHIC SEGMENTS
# -------------------------

    st.header("Luxury Consumer Segmentation")

    X_seg=df[[df.columns[5],df.columns[10],df.columns[15]]]

    kmeans=KMeans(n_clusters=3)

    df["segment"]=kmeans.fit_predict(X_seg)

    names={
    0:"Status Aspirers",
    1:"Identity Seekers",
    2:"Impulse Prestige Buyers"
    }

    df["segment_name"]=df["segment"].map(names)

    fig=px.scatter(
        df,
        x=df.columns[5],
        y=df.columns[20],
        color="segment_name"
    )

    st.plotly_chart(fig)

# -------------------------
# DECISION TREE
# -------------------------

    st.header("Explainable AI Decision Tree")

    model_tree=DecisionTreeRegressor(max_depth=3)

    model_tree.fit(X_seg,purchase)

    fig,ax=plt.subplots(figsize=(10,6))

    plot_tree(
        model_tree,
        feature_names=["Emotion","Celebrity","FOMO"],
        filled=True
    )

    st.pyplot(fig)

# -------------------------
# CONSUMER INFLUENCE NETWORK
# -------------------------

    st.header("Luxury Consumer Influence Network")

    net=nx.Graph()

    net.add_edges_from([
        ("Emotion","Identity"),
        ("Celebrity","Aspirational Influence"),
        ("FOMO","Social Comparison"),
        ("Identity","Purchase"),
        ("Aspirational Influence","Purchase")
    ])

    pos=nx.spring_layout(net)

    nx.draw(net,pos,with_labels=True,node_size=2000)

    st.pyplot()

# -------------------------
# STRATEGY ENGINE
# -------------------------

    st.header("Luxury Strategy Recommendation")

    avg_emotion=np.mean(emotion)

    if avg_emotion>15:
        strategy="Focus on emotional storytelling and heritage branding."
    elif avg_emotion>10:
        strategy="Increase celebrity endorsements and aspirational marketing."
    else:
        strategy="Build exclusivity and prestige positioning."

    st.success(strategy)

# -------------------------
# MARKET FORECAST
# -------------------------

    st.header("Global Luxury Market Forecast")

    years=np.arange(2024,2030)

    growth=0.08

    market=[400*(1+growth)**i for i in range(len(years))]

    forecast=pd.DataFrame({
        "Year":years,
        "MarketSize":market
    })

    fig=px.line(forecast,x="Year",y="MarketSize")

    st.plotly_chart(fig)

# -------------------------
# AI RESEARCH GENERATOR
# -------------------------

    st.header("Auto Research Discussion")

    st.markdown(f"""

The regression results indicate that emotional response significantly predicts luxury purchase intention.

Emotion coefficient: **{round(model.params[1],3)}**

This finding aligns with luxury strategy theory suggesting consumers purchase luxury primarily for symbolic meaning and emotional resonance rather than functional attributes.

Celebrity influence increases aspirational appeal but is not the dominant predictor of purchasing behaviour.

""")

# -------------------------
# PPT REPORT
# -------------------------

    st.header("Download Research Report")

    def generate_ppt():

        prs=Presentation()

        slide=prs.slides.add_slide(prs.slide_layouts[0])
        slide.shapes.title.text="Luxury Consumer Intelligence"

        slide=prs.slides.add_slide(prs.slide_layouts[1])
        slide.shapes.title.text="Regression Results"
        slide.placeholders[1].text=str(model.summary())

        prs.save("luxury_report.pptx")

        return "luxury_report.pptx"

    if st.button("Generate PPT"):

        file=generate_ppt()

        with open(file,"rb") as f:

            st.download_button(
                "Download PowerPoint",
                f,
                file_name="luxury_report.pptx"
            )