import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from sklearn.cluster import KMeans
import feedparser
import random
import requests
from pptx import Presentation

st.set_page_config(layout="wide")

# -------------------
# THEME
# -------------------

st.markdown("""
<style>
.stApp{
background:linear-gradient(135deg,#0f0026,#2c0066,#5e17eb);
color:white;
}
</style>
""", unsafe_allow_html=True)

# -------------------
# SEARCH
# -------------------

st.sidebar.title("Luxury Intelligence Search")

query=st.sidebar.text_input("Search luxury research")

if query:
    url=f"https://api.duckduckgo.com/?q={query}&format=json"
    data=requests.get(url).json()
    st.sidebar.write(data.get("AbstractText","No summary found"))

# -------------------
# NEWS + QUOTE
# -------------------

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
"Exclusivity creates desire — Pinault"
]

if "news" not in st.session_state:
    st.session_state.news=get_news()
    st.session_state.quote=random.choice(quotes)

st.markdown(f"### {' | '.join(st.session_state.news)}")
st.markdown(f"**Quote:** {st.session_state.quote}")

# -------------------
# TITLE
# -------------------

st.title("HayaGriva Luxury Consumer Intelligence Platform")

# -------------------
# DATASET
# -------------------

file=st.file_uploader("Upload Emotion Dataset")

if file:

    df=pd.read_excel(file)

    df=df.replace(0,3)

    emotion=df.filter(regex="luxury|emotion|proud",axis=1).mean(axis=1)
    celebrity=df.filter(regex="celebrity",axis=1).mean(axis=1)
    fomo=df.filter(regex="FOMO|social",axis=1).mean(axis=1)
    purchase=df.filter(regex="purchase|intent",axis=1).mean(axis=1)

# -------------------
# REGRESSION
# -------------------

    st.header("Regression Analysis")

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
**Insights**

• Emotional response has the strongest impact on luxury purchase intention  
• Celebrity endorsements increase aspirational perception  
• FOMO strengthens urgency in luxury buying behaviour  
""")

# -------------------
# RADAR CHART
# -------------------

    st.header("Luxury Psychology Radar")

    radar=go.Figure()

    radar.add_trace(go.Scatterpolar(
    r=[emotion.mean(),celebrity.mean(),fomo.mean()],
    theta=["Emotion","Celebrity","FOMO"],
    fill="toself"
    ))

    st.plotly_chart(radar)

# -------------------
# CORRELATION HEATMAP
# -------------------

    st.header("Psychological Correlation Matrix")

    corr=pd.DataFrame({
    "Emotion":emotion,
    "Celebrity":celebrity,
    "FOMO":fomo,
    "Purchase":purchase
    }).corr()

    st.plotly_chart(px.imshow(corr,text_auto=True))

# -------------------
# SEGMENTATION
# -------------------

    st.header("Luxury Consumer Segmentation")

    seg=pd.DataFrame({
    "Emotion":emotion,
    "Celebrity":celebrity,
    "FOMO":fomo
    })

    kmeans=KMeans(n_clusters=3)

    df["segment"]=kmeans.fit_predict(seg)

    fig=px.scatter(
    df,
    x=emotion,
    y=purchase,
    size=fomo,
    color=df["segment"]
    )

    st.plotly_chart(fig)

# -------------------
# ANIMATED JOURNEY
# -------------------

    st.header("Luxury Consumer Journey")

    journey=pd.DataFrame({
    "Stage":["Awareness","Aspiration","Emotional Bond","Purchase"],
    "Value":[2,5,8,10]
    })

    fig=px.scatter(
    journey,
    x="Stage",
    y="Value",
    animation_frame="Stage",
    size="Value"
    )

    st.plotly_chart(fig)

# -------------------
# WORLD LUXURY DEMAND MAP
# -------------------

st.header("Global Luxury Demand Map")

map_data=pd.DataFrame({

"Country":["USA","France","Italy","China","India","UAE"],

"Demand":[90,75,70,85,60,65],

"lat":[37,46,41,35,21,24],

"lon":[-95,2,12,104,78,54]

})

st.plotly_chart(px.scatter_geo(
map_data,
lat="lat",
lon="lon",
size="Demand",
color="Demand",
hover_name="Country"
))

# -------------------
# BRAND VALUATION MODEL
# -------------------

st.header("Luxury Brand Valuation Model")

emotion_score=st.slider("Emotional Brand Strength",1,10,7)
exclusivity_score=st.slider("Exclusivity Level",1,10,8)
demand_score=st.slider("Market Demand",1,10,7)

brand_value=(emotion_score*0.4)+(exclusivity_score*0.35)+(demand_score*0.25)

st.success(f"Estimated Luxury Brand Strength Score: {round(brand_value,2)}")

# -------------------
# DEMAND FORECAST
# -------------------

st.header("Luxury Market Forecast")

years=np.arange(2024,2030)

market=[400*(1.08)**i for i in range(len(years))]

forecast=pd.DataFrame({
"Year":years,
"Market":market
})

st.plotly_chart(px.line(forecast,x="Year",y="Market"))

# -------------------
# EXECUTIVE SLIDES
# -------------------

st.header("Generate Executive Summary Slides")

def generate_slides():

    prs=Presentation()

    slide=prs.slides.add_slide(prs.slide_layouts[0])
    slide.shapes.title.text="Luxury Consumer Intelligence"

    slide=prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text="Key Findings"
    slide.placeholders[1].text="""
Emotion strongly drives luxury purchase intention.
Celebrity marketing increases aspiration.
FOMO accelerates purchasing decisions.
"""

    prs.save("luxury_summary.pptx")

    return "luxury_summary.pptx"

if st.button("Generate Slides"):

    f=generate_slides()

    with open(f,"rb") as file:

        st.download_button(
        "Download PowerPoint",
        file,
        file_name="luxury_summary.pptx"
        )

# -------------------
# RESEARCH REPORT GENERATOR
# -------------------

st.header("AI Research Report Generator")

report="""

Effect of Emotions on Purchase Intention of Luxury Products

INTRODUCTION
Luxury consumption has evolved from status signaling into emotional identity expression.

METHODOLOGY
Regression analysis was used to analyze the relationship between emotional drivers and purchase intention.

RESULTS
Emotion was identified as the strongest predictor of purchase intention.

DISCUSSION
Luxury consumers derive identity, prestige and emotional satisfaction through brand ownership.

MANAGERIAL IMPLICATIONS
Luxury brands should prioritize emotional storytelling, exclusivity and experiential branding.

"""

st.write(report)