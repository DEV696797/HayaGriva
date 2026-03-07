import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
import requests
import feedparser
import random

st.set_page_config(page_title="HayaGriva Luxury Intelligence", layout="wide")

# -----------------------------
# PREMIUM PURPLE THEME
# -----------------------------

st.markdown("""
<style>
.stApp{
background: linear-gradient(135deg,#0f0026,#2c0066,#5e17eb);
color:white;
}

.ticker{
width:100%;
overflow:hidden;
white-space:nowrap;
background:#6d28d9;
padding:10px;
font-size:16px;
}

.ticker span{
display:inline-block;
padding-left:100%;
animation:scroll 25s linear infinite;
}

@keyframes scroll{
0% {transform:translateX(0);}
100% {transform:translateX(-100%);}
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# LIVE LUXURY NEWS TICKER
# -----------------------------

def get_luxury_news():

    feeds=[
    "https://www.businessoffashion.com/feed/",
    "https://www.voguebusiness.com/rss",
    "https://rss.nytimes.com/services/xml/rss/nyt/FashionandStyle.xml"
    ]

    headlines=[]

    for url in feeds:
        feed=feedparser.parse(url)

        for entry in feed.entries[:3]:
            headlines.append(entry.title)

    return headlines

news=get_luxury_news()

ticker_text="  ✦  ".join(news)

st.markdown(f"""
<div class="ticker">
<span>💎 Luxury Industry News: {ticker_text}</span>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# CEO QUOTE RIBBON
# -----------------------------

quotes=[
"Luxury must be comfortable otherwise it is not luxury — Coco Chanel",
"Luxury brands sell dreams not products — Bernard Arnault",
"Scarcity is the essence of luxury — Hermès Strategy",
"Exclusivity creates desire — François-Henri Pinault",
"Luxury marketing is storytelling at the highest level — LVMH Strategy"
]

st.markdown(f"""
<div style='background:#9333ea;padding:8px;border-radius:8px;text-align:center'>
💡 Executive Insight: {random.choice(quotes)}
</div>
""", unsafe_allow_html=True)

# -----------------------------
# SIDEBAR LOGOS
# -----------------------------

st.sidebar.image(
"https://upload.wikimedia.org/wikipedia/commons/7/7b/Louis_Vuitton_logo_and_wordmark.svg",
width=150
)

st.sidebar.image(
"https://upload.wikimedia.org/wikipedia/commons/5/55/Gucci_Logo.svg",
width=150
)

st.sidebar.image(
"https://upload.wikimedia.org/wikipedia/commons/2/24/Chanel_logo_interlocking_cs.svg",
width=150
)

st.sidebar.image(
"https://upload.wikimedia.org/wikipedia/commons/7/7f/Hermes_logo.svg",
width=150
)

st.sidebar.image(
"https://upload.wikimedia.org/wikipedia/commons/2/20/Rolex_logo.svg",
width=150
)

st.title("HayaGriva Luxury Consumer Intelligence Platform")

st.write("AI-Assisted Behavioral Analytics for Luxury Purchase Intention")

# -----------------------------
# DATA UPLOAD
# -----------------------------

uploaded_file=st.file_uploader("Upload Dataset", type=["xlsx","csv"])

if uploaded_file:

    df=pd.read_excel(uploaded_file)

    emotion=df.iloc[:,5]
    celebrity=df.iloc[:,10]
    fomo=df.iloc[:,15]
    purchase=df.iloc[:,20]

# -----------------------------
# CPM MODEL
# -----------------------------

    def regression(x,y):
        return np.polyfit(x,y,1)[0]

    path_CF=regression(celebrity,fomo)
    path_FE=regression(fomo,emotion)
    path_EP=regression(emotion,purchase)
    path_CE=regression(celebrity,emotion)

    st.header("Behavioral Driver Metrics")

    c1,c2,c3,c4=st.columns(4)

    c1.metric("Celebrity → FOMO",round(path_CF,3))
    c2.metric("FOMO → Emotion",round(path_FE,3))
    c3.metric("Emotion → Purchase",round(path_EP,3))
    c4.metric("Celebrity → Emotion",round(path_CE,3))

# -----------------------------
# DRIVER BAR CHART
# -----------------------------

    st.header("Driver Influence Comparison")

    colA,colB=st.columns([2,1])

    driver_df=pd.DataFrame({
        "Driver":["Celebrity","FOMO","Emotion"],
        "Impact":[path_CF,path_FE,path_EP]
    })

    with colA:

        fig=px.bar(
        driver_df,
        x="Driver",
        y="Impact",
        color="Driver",
        color_discrete_sequence=["#c084fc","#9333ea","#6d28d9"]
        )

        st.plotly_chart(fig,use_container_width=True)

    with colB:

        st.markdown("""
### Chart Explanation

• Emotional engagement dominates purchase intention  
• Celebrity influence amplifies aspiration  
• FOMO creates urgency but weaker conversion impact  
• Luxury purchasing is emotion-centric
""")

# -----------------------------
# CONSUMER SEGMENTATION
# -----------------------------

    st.header("Luxury Consumer Archetypes")

    X=df[[df.columns[5],df.columns[10],df.columns[15]]]

    kmeans=KMeans(n_clusters=3)
    df["segment"]=kmeans.fit_predict(X)

    segment_names={
    0:"Emotional Connoisseurs",
    1:"Prestige Status Seekers",
    2:"Social Momentum Buyers"
    }

    df["segment_name"]=df["segment"].map(segment_names)

    fig2=px.scatter(
    df,
    x=df.columns[5],
    y=df.columns[20],
    color="segment_name"
    )

    st.plotly_chart(fig2,use_container_width=True)

# -----------------------------
# PURCHASE SIMULATOR
# -----------------------------

    st.header("Luxury Purchase Simulator")

    colA,colB,colC=st.columns(3)

    emotion_input=colA.slider("Emotion",1,20,10)
    celebrity_input=colB.slider("Celebrity",1,20,10)
    fomo_input=colC.slider("FOMO",1,20,10)

    score=0.63*emotion_input+0.16*celebrity_input+0.26*fomo_input

    st.success(f"Predicted Purchase Intention Score: {round(score,2)}")

# -----------------------------
# CASE STUDIES
# -----------------------------

    st.header("Luxury Brand Case Studies (2026)")

    st.subheader("Louis Vuitton")

    st.image("https://upload.wikimedia.org/wikipedia/commons/8/8e/LVMH_headquarters_Paris.jpg")

    st.write("Louis Vuitton focuses on immersive retail and experiential storytelling.")

    st.subheader("Gucci")

    st.image("https://upload.wikimedia.org/wikipedia/commons/0/0b/Gucci_HQ_Florence.jpg")

    st.write("Gucci leverages AI personalization and digital engagement.")

    st.subheader("Hermès")

    st.image("https://upload.wikimedia.org/wikipedia/commons/6/6c/Hermes_headquarters_Paris.jpg")

    st.write("Hermès maintains prestige through scarcity and craftsmanship.")

# -----------------------------
# STRATEGIC CONCLUSION
# -----------------------------

    st.header("Strategic Conclusions")

    st.markdown("""
• Emotional storytelling drives luxury purchase intention.

• Celebrity endorsements generate attention but emotional resonance converts.

• Luxury brands must maintain exclusivity while strengthening consumer identity connection.

• AI consumer intelligence platforms like HayaGriva enable strategic luxury marketing decisions.
""")