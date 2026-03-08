import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.express as px

st.set_page_config(page_title="Luxury Consumer Intelligence", layout="wide")

st.title("HayaGriva Luxury Consumer Intelligence")
st.markdown("> *Luxury is the balance of design, beauty and highest quality.* — Domenico De Sole")

st.subheader("Effect of Emotions on Purchase Intention of Luxury Products")

uploaded = st.file_uploader("Upload Dataset", type=["xlsx"])

# Cronbach Alpha Function
def cronbach_alpha(df):

    df_corr = df.corr()

    N = df.shape[1]

    mean_corr = df_corr.values[np.triu_indices(N,1)].mean()

    alpha = (N * mean_corr) / (1 + (N - 1) * mean_corr)

    return alpha

# Item Deleted Alpha
def alpha_if_deleted(df):

    results = []

    for col in df.columns:

        temp = df.drop(columns=[col])

        alpha = cronbach_alpha(temp)

        results.append((col,alpha))

    return pd.DataFrame(results,columns=["Item Removed","Cronbach Alpha"])

if uploaded:

    df = pd.read_excel(uploaded)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    st.subheader("Select Emotion Scale Items (Likert questions)")

    emotion_items = st.multiselect(
        "Select items measuring emotional response",
        numeric_cols
    )

    if len(emotion_items) > 1:

        emotion_df = df[emotion_items]

        alpha = cronbach_alpha(emotion_df)

        st.subheader("Cronbach Alpha Reliability")

        st.metric("Cronbach Alpha", round(alpha,3))

        if alpha >= 0.9:
            st.success("Excellent reliability (SPSS standard)")
        elif alpha >= 0.8:
            st.success("Good reliability")
        elif alpha >= 0.7:
            st.success("Acceptable reliability")
        else:
            st.warning("Scale reliability weak — consider removing problematic items")

        st.subheader("Item Deleted Reliability Table")

        st.dataframe(alpha_if_deleted(emotion_df))

    st.subheader("Regression Analysis")

    X_cols = st.multiselect("Independent Variables", numeric_cols)

    y_col = st.selectbox("Dependent Variable", numeric_cols)

    if len(X_cols)>0 and y_col:

        X = df[X_cols]
        y = df[y_col]

        X_const = sm.add_constant(X)

        model = sm.OLS(y,X_const).fit()

        st.subheader("Regression Results")

        st.text(model.summary())

        r2 = model.rsquared

        st.metric("R²", round(r2,3))

        coef_table = pd.DataFrame({
            "Variable":model.params.index,
            "Coefficient":model.params.values,
            "p-value":model.pvalues.values
        })

        st.subheader("Regression Coefficients")

        st.dataframe(coef_table)

        st.subheader("Hypothesis Testing")

        for var,p in zip(model.params.index,model.pvalues):

            if var!="const":

                if p < 0.05:

                    st.success(f"{var} significantly affects purchase intention (H accepted)")

                else:

                    st.error(f"{var} not statistically significant")

        st.subheader("Regression Equation")

        equation = "Purchase Intention = "

        for i,var in enumerate(model.params.index):

            coef = round(model.params[i],3)

            if var == "const":

                equation += f"{coef}"

            else:

                equation += f" + ({coef} × {var})"

        st.code(equation)

        st.subheader("Correlation Matrix")

        fig = px.imshow(df.corr(), text_auto=True)

        st.plotly_chart(fig)

        st.subheader("Luxury Purchase Simulator")

        inputs = {}

        for col in X_cols:

            inputs[col] = st.slider(col,1,20,10)

        pred = model.predict([[1]+list(inputs.values())])

        st.metric("Predicted Purchase Score",round(pred[0],2))