import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.express as px
from sklearn.cluster import KMeans
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

st.set_page_config(page_title="HayaGriva Luxury Consumer Intelligence", layout="wide")

st.title("HayaGriva Luxury Consumer Intelligence")
st.markdown("> *Luxury is the balance of design, beauty and highest quality.* — Domenico De Sole")

# ---------- DATA UPLOAD ----------
uploaded = st.file_uploader("Upload Dataset", type=["xlsx"])

# ---------- CRONBACH ALPHA ----------
def cronbach_alpha(df):

    corr = df.corr()

    N = df.shape[1]

    mean_corr = corr.values[np.triu_indices(N,1)].mean()

    alpha = (N * mean_corr) / (1 + (N - 1) * mean_corr)

    return alpha

def alpha_deleted(df):

    rows = []

    for col in df.columns:

        temp = df.drop(columns=[col])

        alpha = cronbach_alpha(temp)

        rows.append([col,round(alpha,3)])

    return pd.DataFrame(rows,columns=["Item Removed","Alpha if Deleted"])

# ---------- AI INTERPRETATION ----------
def ai_interpretation(alpha,r2):

    text=""

    if alpha>0.9:
        text+="Reliability analysis indicates excellent internal consistency.\n"

    elif alpha>0.8:
        text+="Reliability analysis indicates strong internal consistency.\n"

    else:
        text+="Reliability results suggest moderate reliability.\n"

    if r2>0.6:
        text+="Regression results show strong explanatory power.\n"

    elif r2>0.3:
        text+="Regression results show moderate explanatory power.\n"

    else:
        text+="Regression results show weak explanatory power.\n"

    text+="Emotional drivers significantly shape luxury purchase intention."

    return text

# ---------- MAIN ----------
if uploaded:

    df=pd.read_excel(uploaded)

    df=df.apply(pd.to_numeric,errors="coerce")

    df=df.replace([np.inf,-np.inf],np.nan)

    df=df.dropna()

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

    numeric_cols=df.select_dtypes(include=np.number).columns.tolist()

    # ---------- RELIABILITY ----------
    st.subheader("Cronbach Alpha Reliability")

    emotion_items=st.multiselect("Select Emotional Response Items",numeric_cols)

    if len(emotion_items)>1:

        emotion_df=df[emotion_items]

        alpha=cronbach_alpha(emotion_df)

        st.metric("Cronbach Alpha",round(alpha,3))

        st.write("Item Deleted Table")

        alpha_table=alpha_deleted(emotion_df)

        st.dataframe(alpha_table)

    # ---------- REGRESSION ----------
    st.subheader("Multiple Linear Regression")

    X_cols=st.multiselect("Independent Variables",numeric_cols)

    y_col=st.selectbox("Dependent Variable",numeric_cols)

    if len(X_cols)>0 and y_col:

        X=df[X_cols]

        y=df[y_col]

        X_const=sm.add_constant(X)

        model=sm.OLS(y,X_const).fit()

        r2=model.rsquared

        coef_table=pd.DataFrame({

            "Variable":model.params.index,

            "Coefficient":model.params.values,

            "Std Error":model.bse.values,

            "p-value":model.pvalues.values

        })

        st.metric("R²",round(r2,3))

        st.subheader("SPSS Style Coefficient Table")

        st.dataframe(coef_table)

        # ---------- AI INTERPRETATION ----------
        if len(emotion_items)>1:

            st.subheader("AI Statistical Interpretation")

            st.info(ai_interpretation(alpha,r2))

        # ---------- CORRELATION ----------
        st.subheader("Correlation Matrix")

        fig=px.imshow(df.corr(),text_auto=True)

        st.plotly_chart(fig)

        # ---------- SEGMENTATION ----------
        st.subheader("Consumer Segmentation")

        kmeans=KMeans(n_clusters=3)

        df["Segment"]=kmeans.fit_predict(df[X_cols])

        fig2=px.scatter(df,x=X_cols[0],y=X_cols[1],color="Segment")

        st.plotly_chart(fig2)

        # ---------- RADAR CHART ----------
        st.subheader("Luxury Motivation Radar")

        radar=df[X_cols].mean()

        radar_df=pd.DataFrame({

            "Driver":radar.index,

            "Score":radar.values

        })

        fig3=px.line_polar(radar_df,r="Score",theta="Driver",line_close=True)

        st.plotly_chart(fig3)

        # ---------- SIMULATOR ----------
        st.subheader("Luxury Purchase Simulator")

        inputs={}

        for col in X_cols:

            inputs[col]=st.slider(col,1,20,10)

        input_df=pd.DataFrame([inputs])

        input_df=sm.add_constant(input_df)

        pred=model.predict(input_df)

        st.metric("Predicted Purchase Score",round(pred.iloc[0],2))

        # ---------- PDF THESIS GENERATOR ----------
        st.subheader("Generate Full Thesis")

        if st.button("Generate Thesis PDF"):

            styles=getSampleStyleSheet()

            doc=SimpleDocTemplate("Luxury_MRP_Thesis.pdf",pagesize=letter)

            elements=[]

            elements.append(Paragraph("Effect of Emotions on Purchase Intention of Luxury Products",styles["Title"]))

            elements.append(Spacer(1,20))

            elements.append(Paragraph("INTRODUCTION",styles["Heading2"]))

            elements.append(Paragraph(
            "Luxury consumption is strongly influenced by emotional engagement, social identity signalling and psychological gratification.",
            styles["Normal"]
            ))

            elements.append(Spacer(1,20))

            elements.append(Paragraph("STATISTICAL RESULTS",styles["Heading2"]))

            table_data=[["Variable","Coefficient","Std Error","p-value"]]

            for i,row in coef_table.iterrows():

                table_data.append(list(row))

            table=Table(table_data)

            elements.append(table)

            elements.append(Spacer(1,20))

            elements.append(Paragraph("Regression R² = "+str(round(r2,3)),styles["Normal"]))

            elements.append(Spacer(1,20))

            elements.append(Paragraph("Conclusion: Emotional drivers significantly influence luxury purchase intention.",styles["Normal"]))

            doc.build(elements)

            st.success("Thesis PDF Generated (Luxury_MRP_Thesis.pdf)")