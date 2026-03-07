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
import feedparser
import random
from pptx import Presentation
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

# -----------------------------
# PREMIUM THEME
# -----------------------------

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

# -----------------------------
# LUXURY NEWS RIBBON
# -----------------------------

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

# -----------------------------
# QUOTE RIBBON
# -----------------------------

quotes=[
"Luxury brands sell dreams not products — Bernard Arnault",
"Luxury is identity not consumption — Kapferer",
"Exclusivity creates desire — Pinault",
"Luxury marketing is storytelling"
]

st.markdown(f"### 💎 Luxury Insight: {random.choice(quotes)}")

# -----------------------------
# SIDEBAR LOGOS
# -----------------------------

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

# -----------------------------
# TITLE
# -----------------------------

st.title("HayaGriva Luxury Consumer Intelligence Platform")

# -----------------------------
# DATA UPLOAD
# -----------------------------

file=st.file_uploader("Upload Luxury Emotion Dataset")

if file:

    df=pd.read_excel(file)

    emotion=df.iloc[:,5]
    celebrity=df.iloc[:,10]
    fomo=df.iloc[:,15]
    purchase=df.iloc[:,20]

# -----------------------------
# MULTIPLE REGRESSION
# -----------------------------

    st.header("Multiple Linear Regression (SPSS Equivalent)")

    X=pd.DataFrame({
        "Emotion":emotion,
        "Celebrity":celebrity,
        "FOMO":fomo
    })

    X=sm.add_constant(X)

    model=sm.OLS(purchase,X).fit()

    st.dataframe(model.summary().tables[1])

    st.metric("R²",round(model.rsquared,3))

# -----------------------------
# SEM MODEL
# -----------------------------

    st.header("Structural Equation Model")

    path_CF=np.corrcoef(celebrity,fomo)[0,1]
    path_FE=np.corrcoef(fomo,emotion)[0,1]
    path_EP=np.corrcoef(emotion,purchase)[0,1]

    G=nx.DiGraph()

    G.add_edge("Celebrity","FOMO")
    G.add_edge("FOMO","Emotion")
    G.add_edge("Emotion","Purchase")

    pos=nx.spring_layout(G)

    edge_x=[]
    edge_y=[]

    for edge in G.edges():
        x0,y0=pos[edge[0]]
        x1,y1=pos[edge[1]]
        edge_x.extend([x0,x1,None])
        edge_y.extend([y0,y1,None])

    fig=go.Figure()

    fig.add_trace(go.Scatter(x=edge_x,y=edge_y,mode='lines'))

    node_x=[]
    node_y=[]

    for node in G.nodes():
        x,y=pos[node]
        node_x.append(x)
        node_y.append(y)

    fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        text=list(G.nodes()),
        textposition="bottom center",
        marker=dict(size=30)
    ))

    st.plotly_chart(fig)

# -----------------------------
# PSYCHOGRAPHIC SEGMENTATION
# -----------------------------

    st.header("Luxury Consumer Segments")

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

# -----------------------------
# DECISION TREE (Explainable AI)
# -----------------------------

    st.header("Explainable AI Decision Tree")

    model_tree=DecisionTreeRegressor(max_depth=3)

    model_tree.fit(X_seg,purchase)

    fig,ax=plt.subplots(figsize=(10,6))

    plot_tree(model_tree,
    feature_names=["Emotion","Celebrity","FOMO"],
    filled=True)

    st.pyplot(fig)

    st.markdown("""
This decision tree shows which psychological factors most strongly drive luxury purchase decisions.
""")

# -----------------------------
# STRATEGY RECOMMENDATION ENGINE
# -----------------------------

    st.header("Luxury Strategy Recommendation Engine")

    avg_emotion=np.mean(emotion)

    if avg_emotion>15:

        strategy="Focus on emotional storytelling and brand heritage campaigns."

    elif avg_emotion>10:

        strategy="Increase celebrity endorsements and aspirational advertising."

    else:

        strategy="Build brand prestige and exclusivity before expanding marketing."

    st.success(strategy)

# -----------------------------
# GLOBAL LUXURY MARKET PREDICTION
# -----------------------------

    st.header("Global Luxury Market Prediction")

    years=np.arange(2024,2030)

    growth=0.08

    market=[400*(1+growth)**i for i in range(len(years))]

    forecast=pd.DataFrame({
        "Year":years,
        "MarketSize":market
    })

    fig=px.line(forecast,x="Year",y="MarketSize")

    st.plotly_chart(fig)

    st.markdown("""
Global luxury market projected to grow strongly driven by emerging markets and digital luxury commerce.
""")

# -----------------------------
# PPT REPORT
# -----------------------------

    st.header("Download Research Report")

    def generate_ppt():

        prs=Presentation()

        slide=prs.slides.add_slide(prs.slide_layouts[0])
        slide.shapes.title.text="HayaGriva Luxury Analytics"

        slide=prs.slides.add_slide(prs.slide_layouts[1])
        slide.shapes.title.text="Regression Results"
        slide.placeholders[1].text=str(model.summary())

        prs.save("luxury_report.pptx")

        return "luxury_report.pptx"

    if st.button("Generate PowerPoint Report"):

        file=generate_ppt()

        with open(file,"rb") as f:

            st.download_button(
                "Download PPT",
                f,
                file_name="luxury_report.pptx"
            )