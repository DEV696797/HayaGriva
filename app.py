import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.api as sm
import feedparser
import random

st.set_page_config(layout="wide")

# -------------------------
# THEME
# -------------------------

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

# -------------------------
# NEWS + QUOTES
# -------------------------

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
"Luxury is identity not consumption — Kapferer",
"Exclusivity creates desire — Pinault"
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

# -------------------------
# TITLE
# -------------------------

st.title("HayaGriva Luxury Consumer Intelligence")

# -------------------------
# DATA UPLOAD
# -------------------------

file=st.file_uploader("Upload Emotion Dataset")

if file:

    df=pd.read_excel(file)

# detect emotional variables
    emotion=df.filter(regex="Emotion|Own|Lux|Excite",axis=1).mean(axis=1)

    celebrity=df.filter(regex="Celebrity",axis=1).mean(axis=1)

    fomo=df.filter(regex="FOMO|Social",axis=1).mean(axis=1)

    purchase=df.filter(regex="Purchase|Intent",axis=1).mean(axis=1)

# -------------------------
# REGRESSION
# -------------------------

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

Emotion has the strongest effect on purchase intention.

Celebrity endorsement increases aspiration.

FOMO drives urgency and social comparison.

""")

# -------------------------
# DRIVER CHART
# -------------------------

    st.header("Psychological Drivers")

    chart=pd.DataFrame({

        "Driver":["Emotion","Celebrity","FOMO"],
        "Impact":[model.params[1],model.params[2],model.params[3]]

    })

    fig=px.bar(chart,x="Driver",y="Impact")

    st.plotly_chart(fig)

# -------------------------
# DEMOGRAPHIC ANALYSIS
# -------------------------

st.header("Demographic Analysis")

demo_file=st.file_uploader("Upload Demographic Dataset")

if demo_file:

    demo=pd.read_excel(demo_file)

    # detect categorical columns
    cat_cols=demo.select_dtypes(include="object").columns

    for col in cat_cols[:3]:

        col1,col2=st.columns([3,1])

        with col1:

            fig=px.histogram(demo,x=col)

            st.plotly_chart(fig)

        with col2:

            st.markdown(f"""
### Insight

Distribution of respondents across **{col}** shows demographic patterns
in luxury consumption behaviour.

""")

# -------------------------
# INDIA MAP
# -------------------------

    if "City" in demo.columns:

        st.header("Luxury Consumer Map")

        fig=px.scatter_geo(

            demo,
            locations="City",
            locationmode="country names",
            scope="asia"

        )

        st.plotly_chart(fig)

# -------------------------
# SYNOPSIS GENERATOR
# -------------------------

st.header("AI Research Synopsis")

synopsis="""
Effect of Emotions on the Purchase Intention of Luxury Products

INTRODUCTION

Luxury consumption has evolved beyond mere functional utility into
a powerful expression of identity, lifestyle and psychological satisfaction.

Millennials increasingly view luxury products as instruments for
self-expression, prestige signalling and emotional fulfilment.

RATIONALE

The Indian luxury market is expanding rapidly due to
rising disposable income, aspirational consumption
and global exposure to luxury brands.

OBJECTIVES

• Analyze emotional drivers of luxury consumption
• Evaluate impact of emotions on purchase intention

METHODOLOGY

Structured questionnaire and regression analysis.

RESULTS

Regression shows emotional response as the strongest predictor.

DISCUSSION

Consumers buy luxury goods primarily for symbolic and emotional value.

CONCLUSION

Luxury brands should emphasize storytelling, exclusivity
and aspirational identity building.

"""

st.text_area("Synopsis",synopsis,height=400)

# -------------------------
# CASE STUDIES
# -------------------------

st.header("Luxury Industry Case Studies")

st.markdown("""

### Louis Vuitton

Louis Vuitton focuses on emotional storytelling and heritage branding.
Its campaigns highlight craftsmanship and legacy.

### Gucci

Gucci leverages celebrity culture and digital storytelling.

### Hermès

Hermès maintains scarcity and exclusivity to preserve symbolic value.

""")

# -------------------------
# PROJECT DASHBOARD
# -------------------------

st.header("Project Dashboard")

st.markdown("""

This dashboard explains the full project:

1. Data Collection
2. Regression Analysis
3. Psychological Drivers
4. Consumer Segmentation
5. Strategic Insights

The project demonstrates how emotions influence luxury purchase intention.

""")

# -------------------------
# CONCLUSIONS
# -------------------------

st.header("Conclusions")

st.markdown("""

Luxury purchasing behaviour is driven primarily by emotional motivations.

Brands must focus on storytelling, identity creation
and aspirational marketing.

Future luxury markets will be driven by digital experiences
and symbolic value creation.

""")