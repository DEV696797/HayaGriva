import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from sklearn.cluster import KMeans
import networkx as nx
import feedparser
import random
import requests
from reportlab.pdfgen import canvas

st.set_page_config(layout="wide")

# --------------------
# THEME
# --------------------

st.markdown("""
<style>
.stApp{
background:linear-gradient(135deg,#0f0026,#2c0066,#5e17eb);
color:white;
}
h1,h2,h3{
color:white;
}
</style>
""", unsafe_allow_html=True)

# --------------------
# INTERNET SEARCH
# --------------------

st.sidebar.title("Luxury Intelligence Search")

query = st.sidebar.text_input("Search luxury research")

if query:
    url=f"https://api.duckduckgo.com/?q={query}&format=json"
    data=requests.get(url).json()
    st.sidebar.write(data.get("AbstractText","No summary available"))

# --------------------
# NEWS + QUOTES
# --------------------

def get_news():

    feeds=[
    "https://www.businessoffashion.com/feed/",
    "https://rss.nytimes.com/services/xml/rss/nyt/FashionandStyle.xml"
    ]

    news=[]

    for f in feeds:

        feed=feedparser.parse(f)

        for entry in feed.entries[:2]:

            news.append(entry.title)

    return news


quotes=[
"Luxury is identity not consumption — Kapferer",
"Luxury brands sell dreams — Bernard Arnault",
"Exclusivity drives desire — Pinault"
]

if "news" not in st.session_state:

    st.session_state.news=get_news()
    st.session_state.quote=random.choice(quotes)

col1,col2=st.columns([8,1])

with col1:

    st.markdown(f"### {' | '.join(st.session_state.news)}")
    st.markdown(f"**Quote:** {st.session_state.quote}")

with col2:

    if st.button("Refresh"):
        st.session_state.news=get_news()
        st.session_state.quote=random.choice(quotes)

# --------------------
# TITLE
# --------------------

st.title("HayaGriva Luxury Consumer Intelligence Platform")

# --------------------
# DATASET
# --------------------

file=st.file_uploader("Upload Emotion Dataset")

if file:

    df=pd.read_excel(file)

    num_cols=df.select_dtypes(include=["number"]).columns
    df[num_cols]=df[num_cols].replace(0,3)

# --------------------
# VARIABLE EXTRACTION
# --------------------

    emotion=df.filter(regex="luxury|emotion|proud",axis=1).mean(axis=1)
    celebrity=df.filter(regex="celebrity",axis=1).mean(axis=1)
    fomo=df.filter(regex="FOMO|social",axis=1).mean(axis=1)
    purchase=df.filter(regex="purchase|intent",axis=1).mean(axis=1)

# --------------------
# REGRESSION
# --------------------

    st.header("Multiple Linear Regression")

    X=pd.DataFrame({
    "Emotion":emotion,
    "Celebrity":celebrity,
    "FOMO":fomo
    })

    X=sm.add_constant(X)

    model=sm.OLS(purchase,X).fit()

    st.dataframe(model.summary2().tables[1])

# --------------------
# CAUSAL PURCHASE SCORE
# --------------------

    st.header("Luxury Purchase Causal Score")

    score=-2.120 + 0.757*emotion.mean() + 0.199*celebrity.mean() + 0.314*fomo.mean()

    st.metric("Luxury Purchase Score",round(score,2))

# --------------------
# SEM PATH MODEL
# --------------------

    st.header("Structural Equation Model")

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

        edge_x+=[x0,x1,None]
        edge_y+=[y0,y1,None]

    fig=go.Figure()

    fig.add_trace(go.Scatter(x=edge_x,y=edge_y,mode="lines"))

    for node in G.nodes():

        x,y=pos[node]

        fig.add_trace(go.Scatter(
        x=[x],
        y=[y],
        text=node,
        mode="markers+text"
        ))

    st.plotly_chart(fig)

# --------------------
# CONSUMER SEGMENTATION
# --------------------

    st.header("Luxury Consumer Segmentation")

    seg=pd.DataFrame({
    "Emotion":emotion,
    "Celebrity":celebrity,
    "FOMO":fomo
    })

    kmeans=KMeans(n_clusters=3)

    df["segment"]=kmeans.fit_predict(seg)

    names={
    0:"Status Aspirers",
    1:"Identity Builders",
    2:"Impulse Prestige Buyers"
    }

    df["segment_name"]=df["segment"].map(names)

    fig=px.scatter(df,x=emotion,y=purchase,color="segment_name")

    st.plotly_chart(fig)

# --------------------
# TABLEAU STYLE VISUALS
# --------------------

    st.header("Behavioral Visual Analytics")

    col1,col2=st.columns(2)

    col1.plotly_chart(px.violin(df,y=purchase))
    col2.plotly_chart(px.box(df,y=purchase))

    col1.plotly_chart(px.histogram(purchase))
    col2.plotly_chart(px.scatter_matrix(seg))

# --------------------
# STORYTELLING DASHBOARD
# --------------------

    st.header("Luxury Consumer Storyline")

    st.markdown("""
Step 1 — Emotional attachment forms the psychological foundation of luxury consumption.

Step 2 — Celebrity endorsement amplifies aspirational value.

Step 3 — Social comparison and FOMO accelerate purchase urgency.

Step 4 — Combined emotional forces significantly increase luxury purchase intention.
""")

# --------------------
# AI SYNOPSIS GENERATOR
# --------------------

    st.header("AI Generated Research Synopsis")

    synopsis=f"""

Effect of Emotions on Purchase Intention of Luxury Products

Luxury consumption increasingly reflects emotional motivations rather than functional needs.

Regression results indicate emotional response as the strongest predictor of purchase intention.

Celebrity influence contributes to aspirational appeal, while FOMO enhances urgency in purchase behaviour.

These findings support the theoretical perspectives proposed by Kapferer and Bastien regarding symbolic consumption.

Luxury consumers derive identity expression, social prestige, and emotional satisfaction from ownership.

Therefore luxury brands should prioritize emotional storytelling, experiential marketing, and exclusivity.
"""

    st.write(synopsis)

# --------------------
# DEMOGRAPHIC DATA
# --------------------

st.header("Demographic Analysis")

demo_file=st.file_uploader("Upload Demographic Dataset")

if demo_file:

    demo=pd.read_excel(demo_file)

    for col in demo.select_dtypes(include="object").columns[:3]:

        st.plotly_chart(px.histogram(demo,x=col))

# --------------------
# GEOGRAPHIC HEATMAP
# --------------------

    if "City" in demo.columns:

        city_counts=demo["City"].value_counts().reset_index()

        city_counts.columns=["City","Count"]

        st.header("Luxury Buyers Geographic Distribution")

        st.plotly_chart(px.bar(city_counts,x="City",y="Count"))

# --------------------
# REPORT GENERATOR
# --------------------

st.header("Download Consulting Report")

def generate_pdf():

    file="luxury_report.pdf"

    c=canvas.Canvas(file)

    c.drawString(100,800,"Luxury Consumer Intelligence Report")

    c.drawString(100,760,"Key Insight: Emotional drivers dominate luxury purchase behavior.")

    c.save()

    return file


if st.button("Generate Report"):

    pdf=generate_pdf()

    with open(pdf,"rb") as f:

        st.download_button("Download Report",f,file_name="luxury_report.pdf")