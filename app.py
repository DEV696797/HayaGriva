import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.api as sm
import feedparser
import random

st.set_page_config(layout="wide")

# --------------------------
# THEME
# --------------------------

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

# --------------------------
# NEWS + QUOTES
# --------------------------

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
"Exclusivity creates desire — François-Henri Pinault"
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

# --------------------------
# TITLE
# --------------------------

st.title("HayaGriva Luxury Consumer Intelligence")

# --------------------------
# DATASET
# --------------------------

file=st.file_uploader("Upload Emotion Dataset")

if file:

    df=pd.read_excel(file)

# Fix Likert Neutral

    num_cols=df.select_dtypes(include=["number"]).columns
    df[num_cols]=df[num_cols].replace(0,3)

    st.info("Neutral responses coded as 0 were recoded to 3 to maintain Likert scale consistency.")

# Detect variables

    emotion=df.filter(regex="Own|emotion|proud|luxury",axis=1).mean(axis=1)
    celebrity=df.filter(regex="celebrity",axis=1).mean(axis=1)
    fomo=df.filter(regex="FOMO|social",axis=1).mean(axis=1)
    purchase=df.filter(regex="purchase|intent",axis=1).mean(axis=1)

# --------------------------
# RELIABILITY
# --------------------------

    st.header("Reliability Analysis")

    def cronbach_alpha(data):

        data=np.array(data)
        itemvars=data.var(axis=0,ddof=1)
        tscores=data.sum(axis=1)

        nitems=data.shape[1]

        alpha=(nitems/(nitems-1))*(1-itemvars.sum()/tscores.var(ddof=1))

        return alpha

    emotion_items=df.filter(regex="luxury|emotion|proud",axis=1)

    st.metric("Emotion Scale Alpha",round(cronbach_alpha(emotion_items),3))

# --------------------------
# REGRESSION
# --------------------------

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

Emotion is the strongest predictor of luxury purchase intention.

Celebrity influence increases aspirational value.

FOMO accelerates purchasing behaviour.
""")

# --------------------------
# DRIVER CHART
# --------------------------

    st.header("Psychological Drivers")

    drivers=pd.DataFrame({

        "Driver":["Emotion","Celebrity","FOMO"],
        "Impact":[model.params[1],model.params[2],model.params[3]]

    })

    fig=px.bar(drivers,x="Driver",y="Impact")

    st.plotly_chart(fig)

# --------------------------
# CAUSAL MODEL
# --------------------------

    st.header("Luxury Purchase Probability Model")

    z=-2.120 + 0.757*emotion.mean() + 0.199*celebrity.mean() + 0.314*fomo.mean()

    prob=1/(1+np.exp(-z))

    st.metric("Probability of Purchasing Luxury Products",f"{round(prob*100,2)}%")

# --------------------------
# SIMULATOR
# --------------------------

    st.subheader("Luxury Purchase Simulator")

    e=st.slider("Emotion Level",1,20,10)
    c=st.slider("Celebrity Influence",1,20,10)
    f=st.slider("FOMO",1,20,10)

    z=-2.120 + 0.757*e + 0.199*c + 0.314*f

    p=1/(1+np.exp(-z))

    st.success(f"Predicted Purchase Probability: {round(p*100,2)}%")

# --------------------------
# DEMOGRAPHICS
# --------------------------

st.header("Demographic Analysis")

demo_file=st.file_uploader("Upload Demographic Dataset")

if demo_file:

    demo=pd.read_excel(demo_file)

    cat_cols=demo.select_dtypes(include="object").columns

    for col in cat_cols[:3]:

        col1,col2=st.columns([3,1])

        with col1:

            fig=px.histogram(demo,x=col)

            st.plotly_chart(fig)

        with col2:

            st.markdown(f"""
Distribution of respondents across **{col}** highlights
demographic patterns in luxury consumption behaviour.
""")

# --------------------------
# CASE STUDIES
# --------------------------

st.header("Luxury Industry Case Studies")

st.markdown("""

### Louis Vuitton – Emotional Storytelling Strategy

Louis Vuitton represents one of the strongest examples of emotional luxury branding. The brand consistently builds narratives around heritage, craftsmanship, and aspirational lifestyles. Instead of emphasizing product functionality, Louis Vuitton campaigns focus on storytelling and emotional connection. Campaigns frequently highlight travel, exploration, and artistic expression, reinforcing the symbolic meaning of luxury.

In relation to this research, Louis Vuitton demonstrates how emotional attachment significantly influences purchase intention. Consumers purchasing Louis Vuitton products often associate them with confidence, status, and personal identity. The brand’s collaborations with artists, celebrities, and designers further strengthen emotional resonance among younger consumers.

The research findings showing a strong relationship between emotional response and purchase intention support Louis Vuitton’s strategy. By activating emotional aspirations rather than focusing solely on product features, the brand sustains strong consumer loyalty and high purchase intention.

---

### Gucci – Celebrity Influence and Cultural Relevance

Gucci has effectively leveraged celebrity culture and pop-culture relevance to expand its luxury appeal. Through collaborations with musicians, actors, and influencers, Gucci positions itself as both aspirational and culturally relevant. Celebrity endorsement campaigns increase visibility and shape consumer perception of the brand as fashionable and desirable.

The regression results in this research demonstrate that celebrity influence positively impacts purchase intention, although its effect is weaker than emotional response. Gucci’s marketing strategy reflects this dynamic: celebrity campaigns create aspirational desire, but emotional identity with the brand ultimately drives purchase behaviour.

Gucci’s use of storytelling through social media platforms further enhances emotional engagement. By combining celebrity influence with strong brand narratives, Gucci maintains high consumer interest among millennials and Gen-Z consumers.

---

### Hermès – Exclusivity and Emotional Prestige

Hermès represents the classic luxury model where exclusivity creates emotional prestige. The brand carefully controls supply, ensuring that products such as the Birkin bag remain scarce and highly desirable. Consumers develop emotional attachment through the perception of rarity and craftsmanship.

The emotional satisfaction associated with owning Hermès products aligns strongly with this research. Consumers derive confidence, identity, and status through luxury ownership. This reinforces the finding that emotional response has the strongest effect on purchase intention.

Hermès demonstrates that luxury brands succeed when they combine emotional symbolism, craftsmanship, and exclusivity to build long-term desirability.
""")

# --------------------------
# CONCLUSIONS
# --------------------------

st.header("Project Conclusions")

st.markdown("""

The findings of this research clearly demonstrate that emotional factors significantly influence the purchase intention of luxury products. Among the independent variables examined, emotional response exhibited the strongest positive effect on purchase intention. Consumers associate luxury products with psychological benefits such as confidence, identity expression, and social recognition.

Celebrity influence and fear of missing out also contribute to luxury purchase behaviour, but their effects are secondary compared to emotional engagement. Celebrity endorsements primarily strengthen aspirational perception, while FOMO accelerates urgency in purchasing decisions.

These results confirm that luxury consumption is fundamentally symbolic and emotionally driven rather than purely functional.

""")

# --------------------------
# FUTURE OUTLOOK
# --------------------------

st.header("Future Outlook for Luxury Markets")

st.markdown("""

The luxury market is expected to grow significantly over the next decade due to increasing global wealth, expanding middle-class populations, and digital transformation. Emerging markets such as India will play a critical role in this expansion.

Luxury brands will increasingly rely on digital storytelling, immersive retail experiences, and personalized consumer engagement to maintain relevance among younger consumers.

Technologies such as artificial intelligence, augmented reality, and digital fashion experiences will further reshape luxury consumption patterns.

""")

# --------------------------
# STRATEGY
# --------------------------

st.header("Strategic Recommendations for Luxury Brands")

st.markdown("""

1. **Prioritize Emotional Storytelling**

Luxury brands should focus on narratives emphasizing heritage, craftsmanship, and identity formation.

2. **Use Celebrity Influence Strategically**

Celebrity collaborations should reinforce aspirational value but remain consistent with brand identity.

3. **Maintain Exclusivity**

Scarcity and limited availability increase emotional desirability and strengthen brand prestige.

4. **Leverage Digital Luxury Experiences**

Interactive digital campaigns and immersive storytelling will attract younger luxury consumers.

""")