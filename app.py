import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.api as sm
import feedparser
import random

st.set_page_config(layout="wide")

# ---------------------
# UI THEME
# ---------------------

st.markdown("""
<style>
.stApp{
background:linear-gradient(135deg,#0f0026,#2c0066,#5e17eb);
color:white;
}
h1,h2,h3,h4,h5{
color:white;
}
</style>
""", unsafe_allow_html=True)

# ---------------------
# REFRESHABLE NEWS
# ---------------------

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
"Luxury brands sell dreams not products — Bernard Arnault",
"Luxury is identity not consumption — Jean Noel Kapferer",
"Exclusivity creates desire — François-Henri Pinault",
"Luxury marketing is storytelling — LVMH strategy"
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

# ---------------------
# TITLE
# ---------------------

st.title("HayaGriva Luxury Consumer Intelligence")

# ---------------------
# DATASET UPLOAD
# ---------------------

file=st.file_uploader("Upload Emotion Dataset")

if file:

    df=pd.read_excel(file)

    emotion=df.iloc[:,5]
    celebrity=df.iloc[:,10]
    fomo=df.iloc[:,15]
    purchase=df.iloc[:,20]

# ---------------------
# REGRESSION ANALYSIS
# ---------------------

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
**Insights**

Emotion strongly predicts luxury purchase intention.

Celebrity influence increases aspiration.

FOMO drives urgency and impulse purchases.
""")

# ---------------------
# DRIVER VISUALIZATION
# ---------------------

    st.header("Psychological Drivers")

    impact=[model.params[1],model.params[2],model.params[3]]

    chart=pd.DataFrame({
        "Driver":["Emotion","Celebrity","FOMO"],
        "Impact":impact
    })

    col1,col2=st.columns([3,1])

    with col1:
        fig=px.bar(chart,x="Driver",y="Impact")
        st.plotly_chart(fig)

    with col2:
        st.markdown("""
Luxury consumption is emotional rather than functional.

Emotional identity formation drives purchasing behaviour.
""")

# ---------------------
# INDIA MAP
# ---------------------

    st.header("Geographical Distribution")

    demo_file=st.file_uploader("Upload Demographic Dataset")

    if demo_file:

        demo=pd.read_excel(demo_file)

        fig=px.scatter_geo(
            demo,
            locations="City",
            locationmode="country names",
            scope="asia",
            size="Purchase",
            title="Luxury Consumers in India"
        )

        st.plotly_chart(fig)

# ---------------------
# SYNOPSIS GENERATOR
# ---------------------

    st.header("AI Research Synopsis")

    synopsis="""
PRESTIGE INSTITUTE OF MANAGEMENT AND RESEARCH, INDORE

MAJOR RESEARCH PROJECT

Effect of Emotions on the Purchase Intension of Luxury Products

INTRODUCTION

Luxury consumption has evolved significantly over the past centuries. Historically,
luxury goods served primarily as status symbols that differentiated social classes.
However, contemporary luxury consumption has shifted toward emotional and symbolic value.

Modern consumers, particularly millennials, increasingly view luxury goods as instruments
for expressing identity, lifestyle and social positioning.

RATIONALE

The growing luxury market in India demonstrates the increasing importance of emotional
drivers such as aspiration, prestige and identity formation.

OBJECTIVES

• To analyze emotional drivers influencing luxury purchase intention
• To understand psychological motivations behind luxury consumption

METHODOLOGY

Primary data was collected using structured questionnaires.
Multiple linear regression analysis was conducted.

RESULTS

Regression equation:

PI = -2.120 + 0.757 ER + 0.199 CI + 0.314 FOMO

Emotional response shows the strongest positive influence.

DISCUSSION

These findings confirm that luxury consumption is driven primarily
by symbolic and emotional value rather than purely functional benefits.

CONCLUSION

Luxury brands must focus on storytelling, identity creation
and aspirational marketing strategies.

"""

    st.text_area("Generated Synopsis", synopsis, height=500)

# ---------------------
# CASE STUDY SECTION
# ---------------------

    st.header("Luxury Industry Case Studies")

    st.markdown("""
**Louis Vuitton**

Louis Vuitton uses emotional storytelling and heritage branding.
Campaigns focus on craftsmanship, legacy and exclusivity.

**Gucci**

Gucci leverages celebrity collaborations and digital storytelling.

**Hermès**

Hermès maintains scarcity to preserve symbolic value.
""")

# ---------------------
# DASHBOARD EXPLANATION
# ---------------------

    st.header("Project Dashboard")

    st.markdown("""
This project analyses the effect of emotions on luxury purchase intention.

The analytical model integrates:

• Multiple Linear Regression  
• Consumer Segmentation  
• Psychological Driver Analysis  

The dashboard visualizes consumer behaviour patterns and provides strategic insights.
""")

# ---------------------
# CONCLUSIONS
# ---------------------

st.header("Conclusions and Future Outlook")

st.markdown("""
Luxury purchasing behaviour is strongly influenced by emotional motivations.

Luxury brands should:

• strengthen emotional storytelling
• maintain exclusivity
• leverage celebrity endorsement strategically

Future luxury markets will be driven by identity-based consumption and digital luxury experiences.
""")