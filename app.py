import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from sklearn.cluster import KMeans
from pptx import Presentation

st.set_page_config(layout="wide")

# ------------------
# THEME
# ------------------

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

# ------------------
# DATA UPLOAD
# ------------------

emotion_file=st.file_uploader("Upload Emotional Dataset")

if emotion_file:

    df=pd.read_excel(emotion_file)

    df=df.replace(0,3)

    emotion=df.iloc[:,5]
    celebrity=df.iloc[:,10]
    fomo=df.iloc[:,15]
    purchase=df.iloc[:,20]

# ------------------
# REGRESSION
# ------------------

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
### Key Insights

• Emotional response has the strongest effect on purchase intention

• Celebrity endorsement enhances aspirational perception

• FOMO drives urgency and social comparison

• Emotional satisfaction is the primary luxury driver
""")

# ------------------
# DRIVER COMPARISON
# ------------------

    st.header("Driver Impact Comparison")

    drivers=pd.DataFrame({
    "Driver":["Emotion","Celebrity","FOMO"],
    "Score":[emotion.mean(),celebrity.mean(),fomo.mean()]
    })

    fig=px.bar(drivers,x="Driver",y="Score")

    st.plotly_chart(fig,use_container_width=True)

# ------------------
# SCATTER RELATIONSHIP
# ------------------

    st.header("Emotion vs Purchase Intention")

    scatter=px.scatter(
    x=emotion,
    y=purchase,
    trendline="ols"
    )

    st.plotly_chart(scatter,use_container_width=True)

# ------------------
# HIGHLIGHT TABLE
# ------------------

    st.header("Psychological Highlight Table")

    matrix=pd.DataFrame({
    "Emotion":emotion,
    "Celebrity":celebrity,
    "FOMO":fomo,
    "Purchase":purchase
    })

    heat=px.imshow(matrix.corr(),text_auto=True)

    st.plotly_chart(heat,use_container_width=True)

# ------------------
# SEGMENTATION
# ------------------

    st.header("Luxury Consumer Segmentation")

    seg=pd.DataFrame({
    "Emotion":emotion,
    "Celebrity":celebrity,
    "FOMO":fomo
    })

    kmeans=KMeans(n_clusters=3)

    df["segment"]=kmeans.fit_predict(seg)

    bubble=px.scatter(
    df,
    x=emotion,
    y=purchase,
    size=fomo,
    color=df["segment"]
    )

    st.plotly_chart(bubble,use_container_width=True)

# ------------------
# 3D PSYCHOLOGICAL MAP
# ------------------

    st.header("Luxury Consumer Psychological Map")

    fig=px.scatter_3d(
    df,
    x=emotion,
    y=celebrity,
    z=fomo,
    size=purchase,
    color=df["segment"]
    )

    st.plotly_chart(fig,use_container_width=True)

# ------------------
# PURCHASE SIMULATOR
# ------------------

st.header("Luxury Purchase Probability Simulator")

emotion_input=st.slider("Emotion",1,20,10)
celebrity_input=st.slider("Celebrity Influence",1,20,10)
fomo_input=st.slider("FOMO",1,20,10)

score=-2.120 + 0.757*emotion_input + 0.199*celebrity_input + 0.314*fomo_input

st.success(f"Predicted Purchase Score: {round(score,2)}")

# ------------------
# PPT GENERATOR
# ------------------

st.header("Generate Gamma-Compatible Slides")

def generate_ppt():

    prs=Presentation()

    slide=prs.slides.add_slide(prs.slide_layouts[0])
    slide.shapes.title.text="Luxury Consumer Analytics"

    slide=prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text="Key Findings"
    slide.placeholders[1].text="""
Emotion strongly drives luxury purchase intention.
Celebrity endorsement builds aspiration.
FOMO increases urgency in buying behaviour.
"""

    slide=prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text="Strategic Implications"
    slide.placeholders[1].text="""
Luxury brands must prioritize emotional storytelling,
identity marketing and exclusivity strategies.
"""

    prs.save("luxury_report.pptx")

    return "luxury_report.pptx"

if st.button("Generate PPT"):

    file=generate_ppt()

    with open(file,"rb") as f:

        st.download_button(
        "Download Presentation",
        f,
        file_name="luxury_report.pptx"
        )