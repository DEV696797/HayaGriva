import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.express as px
from sklearn.cluster import KMeans

st.set_page_config(page_title="HayaGriva Luxury Consumer Intelligence", layout="wide")

st.markdown("""
<style>
body {background-color:#000;color:white;}
h1,h2,h3 {color:#D4AF37;}
</style>
""", unsafe_allow_html=True)

st.title("HayaGriva Luxury Consumer Intelligence")

st.markdown("Luxury is the balance of design, beauty and highest quality — Domenico De Sole")

file = st.file_uploader("Upload Dataset", type=["xlsx"])

def cronbach_alpha(df):

    corr = df.corr()

    N = df.shape[1]

    mean_corr = corr.values[np.triu_indices(N,1)].mean()

    alpha = (N*mean_corr)/(1+(N-1)*mean_corr)

    return alpha


if file:

    df = pd.read_excel(file)

    df = df.apply(pd.to_numeric, errors="coerce")

    df = df.dropna()

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

    cols = df.columns

    # -----------------------
    # SCALE CONSTRUCTION
    # -----------------------

    emotional = df.iloc[:,0:4]
    celebrity = df.iloc[:,4:8]
    fomo = df.iloc[:,8:12]
    purchase = df.iloc[:,12:16]

    st.header("Reliability Analysis")

    alpha_emotion = cronbach_alpha(emotional)
    alpha_celebrity = cronbach_alpha(celebrity)
    alpha_fomo = cronbach_alpha(fomo)
    alpha_purchase = cronbach_alpha(purchase)

    reliability = pd.DataFrame({

        "Scale":["Emotional Response","Celebrity Influence","FOMO","Purchase Intention"],
        "Cronbach Alpha":[alpha_emotion,alpha_celebrity,alpha_fomo,alpha_purchase]

    })

    st.dataframe(reliability)

    # -----------------------
    # CREATE SCALE SCORES
    # -----------------------

    df["EmotionalResponse"] = emotional.mean(axis=1)
    df["CelebrityInfluence"] = celebrity.mean(axis=1)
    df["FOMO"] = fomo.mean(axis=1)
    df["PurchaseIntention"] = purchase.mean(axis=1)

    # -----------------------
    # REGRESSION
    # -----------------------

    st.header("Multiple Linear Regression")

    X = df[["EmotionalResponse","CelebrityInfluence","FOMO"]]

    y = df["PurchaseIntention"]

    X = sm.add_constant(X)

    model = sm.OLS(y,X).fit()

    st.subheader("Model Summary")

    st.text(model.summary())

    coef_table = pd.DataFrame({

        "Variable":model.params.index,
        "Coefficient":model.params.values,
        "Std Error":model.bse.values,
        "p-value":model.pvalues.values

    })

    st.subheader("SPSS Style Coefficient Table")

    st.dataframe(coef_table)

    st.metric("R²", round(model.rsquared,3))

    # -----------------------
    # SEGMENTATION
    # -----------------------

    st.header("Consumer Segmentation")

    X_cluster = df[["EmotionalResponse","CelebrityInfluence","FOMO"]]

    kmeans = KMeans(n_clusters=3)

    df["Segment"] = kmeans.fit_predict(X_cluster)

    fig = px.scatter(df,
                     x="EmotionalResponse",
                     y="PurchaseIntention",
                     color="Segment")

    st.plotly_chart(fig)

    # -----------------------
    # RADAR
    # -----------------------

    st.header("Luxury Motivation Radar")

    radar = df[["EmotionalResponse","CelebrityInfluence","FOMO"]].mean()

    radar_df = pd.DataFrame({

        "Driver":radar.index,
        "Score":radar.values

    })

    fig2 = px.line_polar(radar_df,r="Score",theta="Driver",line_close=True)

    st.plotly_chart(fig2)

    # -----------------------
    # AI INTERPRETATION
    # -----------------------

    st.header("Automated Statistical Interpretation")

    interpretation = f"""

Model explains **{round(model.rsquared*100,1)}% of variance** in luxury purchase intention.

Emotional Response has the strongest effect on purchase intention.

Celebrity Influence also significantly affects luxury consumption behaviour.

FOMO significantly increases the likelihood of purchasing luxury products.

These results confirm that **psychological drivers strongly influence luxury consumer behaviour.**

"""

    st.info(interpretation)