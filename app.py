import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.api as sm
from sklearn.cluster import KMeans
from pptx import Presentation

st.set_page_config(layout="wide")

# -------------------------------------------------
# THEME
# -------------------------------------------------

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

# -------------------------------------------------
# DATA UPLOAD
# -------------------------------------------------

emotion_file = st.file_uploader("Upload Emotional Dataset")
demo_file = st.file_uploader("Upload Demographic Dataset")

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------

if emotion_file:

    df = pd.read_excel(emotion_file)

    df = df.replace(0,3)

    emotion = df.iloc[:,5]
    celebrity = df.iloc[:,10]
    fomo = df.iloc[:,15]
    purchase = df.iloc[:,20]

# -------------------------------------------------
# REGRESSION
# -------------------------------------------------

    st.header("Multiple Linear Regression")

    X = pd.DataFrame({
        "Emotion":emotion,
        "Celebrity":celebrity,
        "FOMO":fomo
    })

    X = sm.add_constant(X)

    model = sm.OLS(purchase,X).fit()

    col1,col2 = st.columns([2,1])

    with col1:
        st.dataframe(model.summary2().tables[1])

    with col2:
        st.markdown("""
### Insights

• Emotional response shows the strongest statistical impact on luxury purchase intention  

• Celebrity endorsements increase aspirational perception  

• FOMO increases urgency and social comparison  

• Psychological drivers strongly explain luxury consumption behaviour
""")

# -------------------------------------------------
# DRIVER COMPARISON
# -------------------------------------------------

    st.header("Psychological Driver Comparison")

    drivers = pd.DataFrame({
        "Driver":["Emotion","Celebrity","FOMO"],
        "Score":[emotion.mean(),celebrity.mean(),fomo.mean()]
    })

    col1,col2 = st.columns([2,1])

    with col1:
        st.plotly_chart(px.bar(drivers,x="Driver",y="Score"),use_container_width=True)

    with col2:
        st.markdown("""
### Interpretation

• Emotional gratification dominates luxury consumption  

• Celebrity endorsements reinforce aspirational identity  

• FOMO increases urgency in luxury purchasing
""")

# -------------------------------------------------
# SCATTER RELATIONSHIP
# -------------------------------------------------

    st.header("Emotion vs Purchase Intention")

    col1,col2 = st.columns([2,1])

    with col1:
        st.plotly_chart(
        px.scatter(x=emotion,y=purchase,trendline="ols"),
        use_container_width=True
        )

    with col2:
        st.markdown("""
### Insights

• Emotional attachment strongly increases purchase intention  

• Consumers with higher emotional scores demonstrate stronger luxury interest  

• Emotional branding is a key driver of luxury demand
""")

# -------------------------------------------------
# SEGMENTATION
# -------------------------------------------------

    st.header("Luxury Consumer Segmentation")

    seg = pd.DataFrame({
        "Emotion":emotion,
        "Celebrity":celebrity,
        "FOMO":fomo
    })

    kmeans = KMeans(n_clusters=3)

    df["segment"] = kmeans.fit_predict(seg)

    col1,col2 = st.columns([2,1])

    with col1:

        bubble = px.scatter(
        df,
        x=emotion,
        y=purchase,
        size=fomo,
        color=df["segment"]
        )

        st.plotly_chart(bubble,use_container_width=True)

    with col2:

        st.markdown("""
### Segments

**Status Aspirers**

• Emotionally driven prestige buyers  

**Celebrity Followers**

• Motivated by endorsements  

**FOMO Buyers**

• Driven by social comparison
""")

# -------------------------------------------------
# DEMOGRAPHICS
# -------------------------------------------------

if demo_file:

    demo = pd.read_excel(demo_file)

    st.header("Demographic Distribution")

    col1,col2 = st.columns([2,1])

    with col1:
        st.plotly_chart(px.histogram(demo,x=demo.columns[0]),use_container_width=True)

    with col2:
        st.markdown("""
### Insights

• Younger consumers demonstrate stronger luxury aspirations  

• Urban consumers display higher purchase intention  

• Exposure to luxury culture increases emotional engagement
""")

# -------------------------------------------------
# PURCHASE SIMULATOR
# -------------------------------------------------

st.header("Luxury Purchase Simulator")

emotion_input = st.slider("Emotion",1,20,10)
celebrity_input = st.slider("Celebrity Influence",1,20,10)
fomo_input = st.slider("FOMO",1,20,10)

score = -2.120 + 0.757*emotion_input + 0.199*celebrity_input + 0.314*fomo_input

st.success(f"Predicted Purchase Score: {round(score,2)}")

# -------------------------------------------------
# CASE STUDIES
# -------------------------------------------------

st.header("Luxury Brand Strategy Case Studies")

st.subheader("Louis Vuitton – Emotional Storytelling Strategy")

st.write("""
Louis Vuitton has consistently relied on emotional storytelling as the foundation of its brand strategy. The company emphasizes heritage, craftsmanship, and artistic collaboration to create strong emotional connections with consumers. Instead of promoting product functionality, Louis Vuitton focuses on symbolic value and aspirational identity.

Campaigns often highlight personal journeys, creativity, and cultural influence. For example, collaborations with artists and global celebrities reinforce the perception that Louis Vuitton represents creativity and cultural prestige. Consumers associate the brand with achievement and personal identity.

This strategy aligns with research findings that emotional attachment significantly increases purchase intention. When consumers perceive luxury brands as extensions of their identity, they become more willing to purchase and display those products.

By emphasizing emotional narratives rather than functional attributes, Louis Vuitton has maintained one of the strongest luxury brand positions globally. This approach demonstrates that emotional resonance plays a central role in luxury consumption behaviour.
""")

st.subheader("Gucci – Aspirational Identity and Celebrity Influence")

st.write("""
Gucci represents a powerful example of how celebrity influence and cultural storytelling can reshape luxury brand perception. Under creative leadership in the last decade, Gucci adopted bold artistic narratives and collaborations with musicians, actors, and influencers.

Celebrity partnerships create aspirational identity signals for consumers. When influential figures adopt luxury brands, consumers perceive those products as symbols of social prestige and personal success. This reinforces emotional desire and increases purchase intention.

Gucci also leverages digital storytelling and social media engagement to strengthen emotional relationships with younger audiences. Campaigns frequently emphasize individuality, creativity, and cultural expression.

The regression results in this research demonstrate that celebrity influence positively contributes to purchase intention. Gucci’s strategy reflects this relationship by using high-visibility cultural figures to enhance brand desirability.

Through emotional storytelling combined with celebrity endorsement, Gucci successfully transformed its brand into a symbol of cultural relevance and aspirational identity.
""")

st.subheader("Hermès – Scarcity and Emotional Desire")

st.write("""
Hermès illustrates how scarcity can intensify emotional desire and reinforce luxury brand value. Unlike many brands that expand production to increase sales, Hermès deliberately restricts supply to maintain exclusivity.

The most famous example is the Birkin bag, which often involves waiting lists and limited availability. This scarcity creates anticipation and emotional excitement among consumers. The difficulty of obtaining the product increases its symbolic value and perceived prestige.

Psychologically, scarcity triggers emotional desire and social comparison. Consumers often interpret rare luxury goods as signals of success and exclusivity. This emotional response increases willingness to pay premium prices.

Research on luxury consumption consistently shows that exclusivity enhances emotional attachment to luxury brands. Hermès successfully leverages this mechanism by controlling distribution and production volume.

The brand’s strategy demonstrates how emotional motivations—particularly prestige and exclusivity—drive luxury purchase intention.
""")

# -------------------------------------------------
# CONCLUSION
# -------------------------------------------------

st.header("Project Conclusion")

st.write("""
The findings of this research confirm that emotional response is the strongest determinant of luxury purchase intention. 
Consumers are motivated not only by product functionality but by psychological gratification, aspirational identity, and symbolic value associated with luxury brands.

Celebrity endorsements and social comparison mechanisms such as FOMO further amplify emotional motivations. These drivers encourage consumers to associate luxury brands with prestige, success, and personal identity.

Luxury companies must therefore focus on emotional storytelling, aspirational positioning, and controlled exclusivity to strengthen consumer attachment and maintain brand desirability.
""")

# -------------------------------------------------
# PPT GENERATOR
# -------------------------------------------------

st.header("Generate Presentation")

def generate_ppt():

    prs = Presentation()

    slide = prs.slides.add_slide(prs.slide_layouts[0])
    slide.shapes.title.text = "Luxury Consumer Research"

    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "Key Findings"
    slide.placeholders[1].text = """
Emotion strongly drives luxury purchase intention.
Celebrity influence increases aspiration.
FOMO accelerates luxury purchase urgency.
"""

    prs.save("luxury_report.pptx")

    return "luxury_report.pptx"

if st.button("Generate PPT"):

    file = generate_ppt()

    with open(file,"rb") as f:
        st.download_button(
            "Download Presentation",
            f,
            file_name="luxury_report.pptx"
        )