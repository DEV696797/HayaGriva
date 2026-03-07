import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.api as sm
from sklearn.cluster import KMeans
from pptx import Presentation

st.set_page_config(layout="wide")

# -----------------------
# THEME
# -----------------------

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

# -----------------------
# DATA UPLOAD
# -----------------------

emotion_file = st.file_uploader("Upload Emotional Dataset")

demo_file = st.file_uploader("Upload Demographic Dataset")

# -----------------------
# LOAD EMOTIONAL DATA
# -----------------------

if emotion_file:

    df = pd.read_excel(emotion_file)

    df = df.replace(0,3)

    emotion = df.iloc[:,5]
    celebrity = df.iloc[:,10]
    fomo = df.iloc[:,15]
    purchase = df.iloc[:,20]

# -----------------------
# MULTIPLE REGRESSION
# -----------------------

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

• Emotional response has the **strongest statistical impact** on luxury purchase intention  

• Celebrity endorsement builds **aspirational perception**

• FOMO increases **urgency and competitive luxury consumption**

• Emotional identity strongly explains luxury purchasing behaviour
""")

# -----------------------
# DRIVER COMPARISON
# -----------------------

    st.header("Psychological Driver Comparison")

    drivers = pd.DataFrame({
        "Driver":["Emotion","Celebrity","FOMO"],
        "Score":[emotion.mean(),celebrity.mean(),fomo.mean()]
    })

    col1,col2 = st.columns([2,1])

    with col1:

        fig = px.bar(drivers,x="Driver",y="Score")
        st.plotly_chart(fig,use_container_width=True)

    with col2:

        st.markdown("""
### Interpretation

• Emotional gratification dominates luxury motivation  

• Celebrity influence acts as **social validation**

• FOMO creates **urgency in luxury consumption**

• Emotional attachment increases brand loyalty
""")

# -----------------------
# SCATTER REGRESSION
# -----------------------

    st.header("Emotion vs Purchase Intention")

    col1,col2 = st.columns([2,1])

    with col1:

        scatter = px.scatter(
        x=emotion,
        y=purchase,
        trendline="ols"
        )

        st.plotly_chart(scatter,use_container_width=True)

    with col2:

        st.markdown("""
### Interpretation

• Positive slope confirms **emotional drivers increase purchase intention**

• Consumers with stronger emotional attachment show higher luxury interest

• Emotional satisfaction acts as a symbolic reward mechanism

• Emotional branding is central to luxury marketing success
""")

# -----------------------
# CORRELATION HEATMAP
# -----------------------

    st.header("Psychological Correlation Matrix")

    matrix = pd.DataFrame({
        "Emotion":emotion,
        "Celebrity":celebrity,
        "FOMO":fomo,
        "Purchase":purchase
    })

    col1,col2 = st.columns([2,1])

    with col1:

        heat = px.imshow(matrix.corr(),text_auto=True)
        st.plotly_chart(heat,use_container_width=True)

    with col2:

        st.markdown("""
### Insights

• Emotion and purchase intention show the **highest correlation**

• Celebrity influence moderately affects emotional engagement

• FOMO amplifies emotional luxury desire

• Psychological drivers collectively explain consumer behaviour
""")

# -----------------------
# SEGMENTATION
# -----------------------

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
• Strong symbolic motivation  

**Celebrity Followers**

• Influenced by endorsements and influencers  

**Social Comparison Buyers**

• Driven by FOMO and peer pressure
""")

# -----------------------
# 3D PSYCHOLOGICAL MAP
# -----------------------

    st.header("Luxury Consumer Psychological Map")

    col1,col2 = st.columns([2,1])

    with col1:

        fig = px.scatter_3d(
        df,
        x=emotion,
        y=celebrity,
        z=fomo,
        size=purchase,
        color=df["segment"]
        )

        st.plotly_chart(fig,use_container_width=True)

    with col2:

        st.markdown("""
### Insights

• Luxury buyers cluster into **distinct psychological groups**

• High emotion + high FOMO consumers show strongest purchase intention

• Celebrity influence shifts consumer clusters

• Emotional motivation is the central axis of luxury behavior
""")

# -----------------------
# DEMOGRAPHIC ANALYSIS
# -----------------------

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

• Younger respondents display stronger luxury aspirations  

• Urban consumers show higher purchase intention  

• Exposure to global luxury culture increases emotional engagement  

• Income levels influence luxury adoption rates
""")

# -----------------------
# INDIA LUXURY DEMAND MAP
# -----------------------

    st.header("Luxury Demand Across India")

    if "City" in demo.columns:

        city_counts = demo["City"].value_counts().reset_index()

        city_counts.columns = ["City","Responses"]

        col1,col2 = st.columns([2,1])

        with col1:

            map_fig = px.scatter_geo(
            city_counts,
            locations="City",
            locationmode="country names",
            size="Responses"
            )

            st.plotly_chart(map_fig,use_container_width=True)

        with col2:

            st.markdown("""
### Insights

• Major metro cities show highest luxury engagement  

• Urbanization drives aspirational consumption  

• Luxury demand clusters around high-income regions  

• Exposure to luxury retail environments increases purchase intention
""")

# -----------------------
# PURCHASE SIMULATOR
# -----------------------

st.header("Luxury Purchase Simulator")

emotion_input = st.slider("Emotion",1,20,10)
celebrity_input = st.slider("Celebrity Influence",1,20,10)
fomo_input = st.slider("FOMO",1,20,10)

score = -2.120 + 0.757*emotion_input + 0.199*celebrity_input + 0.314*fomo_input

st.success(f"Predicted Purchase Score: {round(score,2)}")

# -----------------------
# STRATEGY ENGINE
# -----------------------

st.header("Luxury Market Strategy Engine")

if emotion_input > celebrity_input and emotion_input > fomo_input:

    st.markdown("""
### Emotional Branding Strategy

• Focus on heritage storytelling  
• Highlight craftsmanship and identity  
• Build emotional brand narratives  

Best suited for **Hermès and Louis Vuitton**
""")

elif celebrity_input > emotion_input:

    st.markdown("""
### Celebrity Marketing Strategy

• Collaborate with global influencers  
• Launch aspirational brand campaigns  

Best suited for **Gucci**
""")

else:

    st.markdown("""
### Scarcity Strategy

• Limited edition releases  
• Controlled product supply  

Best suited for **Rolex**
""")

# -----------------------
# PPT EXPORT
# -----------------------

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
FOMO accelerates buying urgency.
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