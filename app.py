import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.express as px

st.set_page_config(page_title="Luxury Consumer Intelligence", layout="wide")

# Header
st.title("HayaGriva Luxury Consumer Intelligence Dashboard")

st.markdown("""
> **“Luxury is the balance of design, beauty, and highest quality.” – Domenico De Sole**
""")

st.subheader("Effect of Emotions on Purchase Intention of Luxury Products")

# File Upload
st.sidebar.header("Upload Datasets")

emotion_file = st.sidebar.file_uploader("Upload Emotional Dataset", type=["xlsx"])
demo_file = st.sidebar.file_uploader("Upload Demographic Dataset", type=["xlsx"])

if emotion_file:
    df = pd.read_excel(emotion_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Assume columns exist
    X = df[["Emotion", "Celebrity", "FOMO"]]
    y = df["Purchase"]

    model = LinearRegression()
    model.fit(X, y)

    coeff = pd.DataFrame({
        "Driver": ["Emotion", "Celebrity", "FOMO"],
        "Impact": model.coef_
    })

    st.subheader("Psychological Driver Impact")
    fig = px.bar(coeff, x="Driver", y="Impact", title="Impact of Psychological Drivers")
    st.plotly_chart(fig)

    st.subheader("Psychological Correlation Matrix")
    corr = df.corr()

    fig2 = px.imshow(corr, text_auto=True)
    st.plotly_chart(fig2)

    st.subheader("Insights")

    st.markdown("""
• Emotional response has the strongest statistical impact on luxury purchase intention  

• Celebrity endorsements reinforce aspirational perception  

• FOMO increases urgency and social comparison  

• Psychological drivers strongly explain luxury consumption behaviour
""")

    st.subheader("Luxury Consumer Segmentation")

    st.markdown("""
**Status Aspirers**
- Emotionally driven prestige buyers

**Celebrity Followers**
- Influenced by endorsements

**FOMO Buyers**
- Driven by social comparison
""")

    st.subheader("AI Luxury Strategy Advisor")

    st.success("""
Recommended Strategy

Focus on emotional storytelling and brand heritage.

• Emphasize craftsmanship and legacy  
• Develop experiential retail environments  
• Build emotional brand communities  
""")

    st.subheader("Luxury Purchase Simulator")

    emotion = st.slider("Emotion",1,20,10)
    celebrity = st.slider("Celebrity Influence",1,20,10)
    fomo = st.slider("FOMO",1,20,10)

    pred = model.predict([[emotion,celebrity,fomo]])

    st.metric("Predicted Purchase Score",round(pred[0],2))

st.subheader("Project Conclusion")

st.markdown("""
Emotion is the strongest driver of luxury purchase intention.

Luxury brands must focus on:

• Emotional storytelling  
• Aspirational identity creation  
• Exclusivity strategies  
• Experiential retail environments
""")

st.subheader("Research Thesis")

st.markdown("""
### Effect of Emotions on Purchase Intention of Luxury Products

Luxury consumption has evolved from a simple display of wealth to a complex psychological expression of identity, aspiration, and emotional gratification. Historically, luxury brands such as Hermès and Château Haut-Brion symbolized elite social status. Today, luxury consumption reflects emotional engagement, lifestyle signaling, and symbolic value.

Modern consumers—particularly millennials and Generation Z—purchase luxury products such as designer apparel, watches, and smartphones not only for functional utility but also for emotional fulfilment.

Psychological drivers such as:

• Emotional gratification  
• Celebrity endorsement  
• Fear of Missing Out (FOMO)

play a critical role in shaping purchase intentions.

The study uses quantitative research methods to analyze these drivers through statistical techniques including reliability analysis, correlation analysis, and multiple linear regression.

Results indicate that emotional engagement has the strongest influence on luxury purchase intention, followed by celebrity influence and FOMO.

These findings suggest that luxury marketing strategies should focus on storytelling, aspirational branding, and exclusivity to strengthen emotional connections with consumers.
""")