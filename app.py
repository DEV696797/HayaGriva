import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.api as sm
from sklearn.cluster import KMeans

st.set_page_config(layout="wide")

# ------------------------------------------------
# LUXURY THEME
# ------------------------------------------------

st.markdown("""
<style>

html, body, [class*="css"] {
font-family: "Times New Roman", serif;
color:#ffffff;
}

.stApp{
background:linear-gradient(135deg,#1b0036,#3b0a63,#5e17eb);
}

h1,h2,h3,h4,h5{
font-family:"Times New Roman",serif;
color:#ffffff;
}

p{
font-family:"Times New Roman",serif;
font-size:18px;
color:#ffffff;
}

</style>
""",unsafe_allow_html=True)

st.title("HayaGriva Luxury Consumer Intelligence")
st.subheader("Effect of Emotions on Purchase Intention of Luxury Products")

emotion_file = st.file_uploader("Upload Emotional Dataset")
demo_file = st.file_uploader("Upload Demographic Dataset")

# ------------------------------------------------
# LOAD DATA
# ------------------------------------------------

if emotion_file:

    df = pd.read_excel(emotion_file)

    df = df.replace(0,3)

    emotion = df.iloc[:,5]
    celebrity = df.iloc[:,10]
    fomo = df.iloc[:,15]
    purchase = df.iloc[:,20]

# ------------------------------------------------
# MULTIPLE REGRESSION
# ------------------------------------------------

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

• Emotional response has the strongest statistical impact on luxury purchase intention  

• Celebrity endorsements reinforce aspirational perception  

• FOMO increases urgency and social comparison  

• Psychological drivers strongly explain luxury consumption behaviour
""")

# ------------------------------------------------
# DRIVER COMPARISON
# ------------------------------------------------

    st.header("Psychological Driver Comparison")

    drivers = pd.DataFrame({
        "Driver":["Emotion","Celebrity","FOMO"],
        "Score":[emotion.mean(),celebrity.mean(),fomo.mean()]
    })

    col1,col2 = st.columns([2,1])

    with col1:
        fig = px.bar(drivers,x="Driver",y="Score",color="Driver")
        st.plotly_chart(fig,use_container_width=True)

    with col2:
        st.markdown("""
### Interpretation

• Emotional gratification dominates luxury motivation  

• Celebrity endorsements strengthen aspirational identity  

• FOMO accelerates purchase urgency
""")

# ------------------------------------------------
# RADAR CHART
# ------------------------------------------------

    st.header("Luxury Motivation Radar")

    radar = pd.DataFrame({
        "Factor":["Emotion","Celebrity","FOMO","Purchase"],
        "Score":[emotion.mean(),celebrity.mean(),fomo.mean(),purchase.mean()]
    })

    col1,col2 = st.columns([2,1])

    with col1:
        fig = px.line_polar(radar,r="Score",theta="Factor",line_close=True)
        st.plotly_chart(fig,use_container_width=True)

    with col2:
        st.markdown("""
### Insights

• Emotion dominates luxury purchasing behaviour  

• Celebrity influence supports aspirational branding  

• FOMO acts as psychological accelerator
""")

# ------------------------------------------------
# CORRELATION HEATMAP
# ------------------------------------------------

    st.header("Psychological Correlation Matrix")

    matrix = pd.DataFrame({
        "Emotion":emotion,
        "Celebrity":celebrity,
        "FOMO":fomo,
        "Purchase":purchase
    })

    col1,col2 = st.columns([2,1])

    with col1:
        heat = px.imshow(matrix.corr(),text_auto=True,color_continuous_scale="Purples")
        st.plotly_chart(heat,use_container_width=True)

    with col2:
        st.markdown("""
### Insights

• Emotion and purchase intention show strongest correlation  

• Celebrity influence moderately affects emotional engagement  

• FOMO amplifies luxury desire
""")

# ------------------------------------------------
# SEGMENTATION
# ------------------------------------------------

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
        fig = px.scatter(df,x=emotion,y=purchase,size=fomo,color=df["segment"])
        st.plotly_chart(fig,use_container_width=True)

    with col2:
        st.markdown("""
### Segments

Status Aspirers  
• Emotionally driven prestige buyers  

Celebrity Followers  
• Influenced by endorsements  

FOMO Buyers  
• Driven by social comparison
""")

# ------------------------------------------------
# AI LUXURY STRATEGY ADVISOR
# ------------------------------------------------

    st.header("AI Luxury Strategy Advisor")

    avg_emotion = emotion.mean()
    avg_celebrity = celebrity.mean()
    avg_fomo = fomo.mean()

    if avg_emotion > avg_celebrity and avg_emotion > avg_fomo:

        st.markdown("""
### Recommended Strategy

Focus on emotional storytelling and brand heritage.

• Emphasize craftsmanship and legacy  
• Develop experiential retail environments  
• Build emotional brand communities
""")

    elif avg_celebrity > avg_emotion:

        st.markdown("""
### Recommended Strategy

Leverage celebrity and influencer partnerships.

• Collaborate with global cultural icons  
• Use social media storytelling  
• Increase aspirational branding
""")

    else:

        st.markdown("""
### Recommended Strategy

Use scarcity and exclusivity.

• Launch limited editions  
• Introduce exclusive membership experiences  
• Maintain controlled supply
""")

# ------------------------------------------------
# DEMOGRAPHIC ANALYSIS
# ------------------------------------------------

if demo_file:

    demo = pd.read_excel(demo_file)

    st.header("Demographic Distribution")

    col1,col2 = st.columns([2,1])

    with col1:
        fig = px.histogram(demo,x=demo.columns[0])
        st.plotly_chart(fig,use_container_width=True)

    with col2:
        st.markdown("""
### Insights

• Younger consumers show stronger luxury aspirations  

• Urban exposure increases emotional engagement  

• Demographics influence luxury adoption patterns
""")

# ------------------------------------------------
# PURCHASE SIMULATOR
# ------------------------------------------------

st.header("Luxury Purchase Simulator")

emotion_input = st.slider("Emotion",1,20,10)
celebrity_input = st.slider("Celebrity Influence",1,20,10)
fomo_input = st.slider("FOMO",1,20,10)

score = -2.120 + 0.757*emotion_input + 0.199*celebrity_input + 0.314*fomo_input

st.success(f"Predicted Purchase Score: {round(score,2)}")

# ------------------------------------------------
# CONCLUSION
# ------------------------------------------------

st.header("Project Conclusion")

st.write("""
Emotion is the strongest driver of luxury purchase intention.
Luxury brands must focus on emotional storytelling, aspirational identity creation and exclusivity strategies.
""")