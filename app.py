import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

st.set_page_config(page_title="Luxury Consumer Intelligence", layout="wide")

st.markdown(
"""
<style>
body{
background-color:#0f0f0f;
color:white;
}
h1,h2,h3{
color:#d4af37;
}
</style>
""",
unsafe_allow_html=True
)

st.title("HayaGriva Luxury Consumer Intelligence")

st.markdown("> *Luxury is the balance of design, beauty and highest quality.* — Domenico De Sole")

st.subheader("Effect of Emotions on Purchase Intention of Luxury Products")

# Upload dataset
uploaded_file = st.file_uploader("Upload Dataset", type=["xlsx"])

def cronbach_alpha(df):
    df_corr = df.corr()
    N = len(df.columns)
    mean_corr = df_corr.values[np.triu_indices(N,1)].mean()
    alpha = (N * mean_corr) / (1 + (N-1) * mean_corr)
    return alpha

if uploaded_file:

    df = pd.read_excel(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    st.subheader("Select Variables")

    X_cols = st.multiselect("Independent Variables", numeric_cols)
    y_col = st.selectbox("Dependent Variable", numeric_cols)

    if len(X_cols)>0 and y_col:

        X = df[X_cols]
        y = df[y_col]

        # Cronbach Alpha
        alpha = cronbach_alpha(X)

        st.subheader("Cronbach Alpha Reliability Test")

        st.metric("Cronbach Alpha", round(alpha,3))

        if alpha > 0.7:
            st.success("Scale is reliable")
        else:
            st.warning("Scale reliability is weak")

        # Regression
        X_const = sm.add_constant(X)
        model = sm.OLS(y,X_const).fit()

        st.subheader("Regression Summary")

        st.text(model.summary())

        r2 = model.rsquared

        st.metric("R² Value", round(r2,3))

        st.subheader("Regression Coefficients")

        coef_table = pd.DataFrame({
            "Variable":model.params.index,
            "Coefficient":model.params.values,
            "p-value":model.pvalues.values
        })

        st.dataframe(coef_table)

        # Hypothesis testing
        st.subheader("Hypothesis Testing")

        for var,p in zip(model.params.index,model.pvalues):
            if var!="const":
                if p<0.05:
                    st.success(f"{var} significantly affects purchase intention (H accepted)")
                else:
                    st.error(f"{var} not significant (H rejected)")

        # Regression equation
        st.subheader("Regression Equation")

        equation = "Purchase = "

        for i,var in enumerate(model.params.index):
            coef = round(model.params[i],3)

            if var=="const":
                equation += f"{coef}"
            else:
                equation += f" + ({coef} × {var})"

        st.code(equation)

        # Correlation matrix
        st.subheader("Correlation Matrix")

        fig = px.imshow(df.corr(), text_auto=True)

        st.plotly_chart(fig)

        # Simulator
        st.subheader("Luxury Purchase Simulator")

        inputs={}

        for col in X_cols:
            inputs[col]=st.slider(col,1,20,10)

        pred = model.predict([ [1] + list(inputs.values()) ])

        st.metric("Predicted Purchase Score",round(pred[0],2))

# Thesis section
st.subheader("Research Report")

st.markdown("""

### INTRODUCTION

Luxury products have evolved from mere indicators of wealth into powerful symbols of identity, lifestyle, and emotional gratification. Historically, luxury brands such as Hermès and Château Haut-Brion established the foundation of prestige consumption through craftsmanship, exclusivity, and heritage.

Today luxury consumption is deeply influenced by psychological drivers such as emotional gratification, celebrity endorsement, and social comparison.

Millennial consumers often purchase luxury goods not only for functional benefits but also for symbolic value, social recognition, and emotional satisfaction.

### OBJECTIVES

• To identify factors influencing luxury purchase intention  
• To examine emotional impact on luxury consumption behaviour

### METHODOLOGY

The research employs a quantitative design using structured questionnaires.

Data is analyzed using:

• Cronbach Alpha reliability testing  
• Correlation analysis  
• Multiple linear regression

### FINDINGS

Regression results indicate that emotional engagement significantly predicts luxury purchase intention.

The R² statistic explains how much variance in purchase behaviour is explained by emotional drivers.

### MANAGERIAL IMPLICATIONS

Luxury brands should focus on:

• Emotional storytelling  
• Celebrity partnerships  
• Scarcity marketing  
• Experiential retail environments

""")

# PDF export
if st.button("Export Thesis PDF"):

    styles = getSampleStyleSheet()

    doc = SimpleDocTemplate("luxury_thesis.pdf", pagesize=letter)

    story=[]

    story.append(Paragraph("Effect of Emotions on Purchase Intention of Luxury Products",styles["Title"]))

    story.append(Spacer(1,20))

    story.append(Paragraph("Luxury consumption is strongly influenced by emotional engagement, social status signaling and psychological motivations.",styles["Normal"]))

    doc.build(story)

    st.success("PDF Generated: luxury_thesis.pdf")