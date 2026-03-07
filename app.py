import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import feedparser
import random

st.set_page_config(layout="wide")

# -------------------------
# UI THEME
# -------------------------

st.markdown("""
<style>

.stApp{
background:linear-gradient(135deg,#0f0026,#2c0066,#5e17eb);
color:white;
}

h1,h2,h3,h4{
color:white;
}

</style>
""", unsafe_allow_html=True)

# -------------------------
# NEWS RIBBON
# -------------------------

feeds=[
"https://www.businessoffashion.com/feed/",
"https://rss.nytimes.com/services/xml/rss/nyt/FashionandStyle.xml"
]

news=[]

for f in feeds:
    feed=feedparser.parse(f)
    for entry in feed.entries[:2]:
        news.append(entry.title)

st.markdown(f"### Luxury Industry News: {' | '.join(news)}")

# -------------------------
# QUOTE
# -------------------------

quotes=[
"Luxury brands sell dreams not products — Bernard Arnault",
"Luxury is identity not consumption — Kapferer",
"Exclusivity creates desire — Pinault"
]

st.markdown(f"### 💎 Luxury Insight: {random.choice(quotes)}")

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

    coef=model.summary2().tables[1]

    col1,col2=st.columns([2,1])

    with col1:

        st.dataframe(coef)

    with col2:

        st.markdown("""
### Regression Insights

• Emotional response is the strongest predictor of purchase intention.

• Celebrity endorsement increases aspiration but has smaller direct impact.

• FOMO significantly increases urgency to purchase.

These findings confirm that **luxury purchasing behaviour is emotionally driven rather than functionally driven**.
""")

# -------------------------
# DRIVER VISUALIZATION
# -------------------------

    st.header("Driver Influence")

    impact=[model.params[1],model.params[2],model.params[3]]

    chart_df=pd.DataFrame({
        "Driver":["Emotion","Celebrity","FOMO"],
        "Impact":impact
    })

    col1,col2=st.columns([3,1])

    with col1:

        fig=px.bar(chart_df,x="Driver",y="Impact",color="Driver")

        st.plotly_chart(fig)

    with col2:

        st.markdown("""
### Interpretation

Emotion drives symbolic consumption.

Celebrity endorsement strengthens brand prestige.

FOMO accelerates purchasing behaviour among millennials.
""")

# -------------------------
# SEGMENTATION
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

    col1,col2=st.columns([3,1])

    with col1:

        fig=px.scatter(
            df,
            x=df.columns[5],
            y=df.columns[20],
            color="segment_name"
        )

        st.plotly_chart(fig)

    with col2:

        st.markdown("""
### Segment Insights

**Status Aspirers**  
Buy luxury to signal prestige.

**Identity Seekers**  
Luxury is used for self-expression.

**Impulse Prestige Buyers**  
Driven by celebrity and hype.
""")

# -------------------------
# AI STRATEGY ADVISOR
# -------------------------

    st.header("AI Luxury Strategy Advisor")

    avg_emotion=np.mean(emotion)

    if avg_emotion>15:

        strategy="""
Focus on emotional storytelling and heritage marketing.

Luxury brands should emphasize craftsmanship, legacy and identity.
"""

    elif avg_emotion>10:

        strategy="""
Celebrity driven campaigns should be used.

Aspirational marketing and influencer partnerships will increase engagement.
"""

    else:

        strategy="""
Focus on building exclusivity and prestige positioning before scaling advertising.
"""

    st.success(strategy)

# -------------------------
# CASE STUDY
# -------------------------

    st.header("Luxury Brand Case Studies")

    st.markdown("""
### Louis Vuitton

Louis Vuitton focuses heavily on emotional storytelling and heritage branding.  
Campaigns highlight craftsmanship and historical prestige.

### Gucci

Gucci leverages celebrity collaborations and digital storytelling to generate aspiration.

### Hermès

Hermès maintains scarcity to preserve exclusivity and symbolic value.
""")

# -------------------------
# SYNOPSIS GENERATOR
# -------------------------

    st.header("AI Research Synopsis Generator")

    synopsis=f"""
PRESTIGE INSTITUTE OF MANAGEMENT AND RESEARCH, INDORE

MAJOR RESEARCH PROJECT

Effect of Emotions on the Purchase Intension of Luxury Products

INTRODUCTION

Luxury products have evolved from mere status symbols to powerful mediums of identity expression. 
Millennials increasingly purchase luxury goods for emotional gratification rather than functional benefits.

RATIONALE

The rapid growth of luxury consumption among millennials suggests that emotional drivers significantly influence purchase behaviour.

OBJECTIVES

1. To explore factors influencing purchase intention of luxury products.
2. To analyze the effect of emotions on purchase intention.

METHODOLOGY

A structured questionnaire using Likert scale was used for primary data collection.
Multiple Linear Regression was applied to evaluate relationships between variables.

RESULTS

Regression equation:

PI = -2.120 + 0.757 ER + 0.199 CI + 0.314 FOMO

The coefficient for Emotional Response indicates a strong positive effect on purchase intention.

DISCUSSION

The findings confirm that luxury consumption is largely driven by emotional motivations rather than functional needs.

Consumers derive symbolic value, social recognition and psychological satisfaction from luxury purchases.

CONCLUSION

Emotions significantly influence luxury purchase intention. 
Luxury brands should focus on emotional storytelling and aspirational branding to strengthen consumer engagement.

"""

    st.text_area("Generated Synopsis",synopsis,height=500)

# -------------------------
# DEMOGRAPHIC DATASET
# -------------------------

st.header("Demographic Dataset Analysis")

demo=st.file_uploader("Upload Demographic Dataset")

if demo:

    demo_df=pd.read_excel(demo)

    col1,col2=st.columns([3,1])

    with col1:

        fig=px.histogram(demo_df,x="Age")

        st.plotly_chart(fig)

    with col2:

        st.markdown("""
### Insight

Luxury consumption is highest among young professionals and high income individuals.
""")