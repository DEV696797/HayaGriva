import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.express as px
from factor_analyzer.factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
from sklearn.cluster import KMeans

st.set_page_config(page_title="Luxury Consumer Intelligence", layout="wide")

st.title("HayaGriva Luxury Consumer Intelligence")
st.markdown("> Luxury is the balance of design, beauty and highest quality. — Domenico De Sole")

uploaded = st.file_uploader("Upload Dataset", type=["xlsx"])

# Cronbach Alpha
def cronbach_alpha(df):

    corr = df.corr()

    N = df.shape[1]

    mean_corr = corr.values[np.triu_indices(N,1)].mean()

    alpha = (N * mean_corr) / (1 + (N - 1) * mean_corr)

    return alpha

# Item deleted alpha
def alpha_deleted(df):

    results = []

    for col in df.columns:

        temp = df.drop(columns=[col])

        alpha = cronbach_alpha(temp)

        results.append((col,alpha))

    return pd.DataFrame(results,columns=["Item Removed","Alpha"])

if uploaded:

    df = pd.read_excel(uploaded)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    # Reliability
    st.subheader("Cronbach Alpha Reliability")

    emotion_items = st.multiselect(
        "Select Emotion Scale Items",
        numeric_cols
    )

    if len(emotion_items) > 1:

        emotion_df = df[emotion_items]

        alpha = cronbach_alpha(emotion_df)

        st.metric("Cronbach Alpha", round(alpha,3))

        st.write("Item Deleted Reliability")

        st.dataframe(alpha_deleted(emotion_df))

    # KMO + Bartlett
    st.subheader("Factor Analysis Diagnostics")

    if len(emotion_items) > 1:

        kmo_all, kmo_model = calculate_kmo(emotion_df)

        chi_square_value,p_value = calculate_bartlett_sphericity(emotion_df)

        st.metric("KMO Measure", round(kmo_model,3))

        st.metric("Bartlett p-value", round(p_value,5))

    # Factor Analysis
    st.subheader("Exploratory Factor Analysis")

    if len(emotion_items) > 1:

        fa = FactorAnalyzer()

        fa.fit(emotion_df)

        ev, v = fa.get_eigenvalues()

        eigen_df = pd.DataFrame({"Eigenvalue":ev})

        st.line_chart(eigen_df)

    # Regression
    st.subheader("Multiple Linear Regression")

    X_cols = st.multiselect("Independent Variables", numeric_cols)

    y_col = st.selectbox("Dependent Variable", numeric_cols)

    if len(X_cols)>0 and y_col:

        X = df[X_cols]

        y = df[y_col]

        X_const = sm.add_constant(X)

        model = sm.OLS(y,X_const).fit()

        st.text(model.summary())

        st.metric("R²", round(model.rsquared,3))

        coef_table = pd.DataFrame({
            "Variable":model.params.index,
            "Coefficient":model.params.values,
            "p-value":model.pvalues.values
        })

        st.dataframe(coef_table)

        # Hypothesis
        st.subheader("Hypothesis Testing")

        for var,p in zip(model.params.index,model.pvalues):

            if var!="const":

                if p < 0.05:

                    st.success(f"{var} significantly affects purchase intention")

                else:

                    st.error(f"{var} not statistically significant")

        # Regression Equation
        st.subheader("Regression Equation")

        eq = "Purchase Intention = "

        for i,var in enumerate(model.params.index):

            coef = round(model.params[i],3)

            if var=="const":

                eq += str(coef)

            else:

                eq += f" + ({coef} × {var})"

        st.code(eq)

        # Correlation
        st.subheader("Correlation Matrix")

        fig = px.imshow(df.corr(), text_auto=True)

        st.plotly_chart(fig)

        # Segmentation
        st.subheader("Consumer Segmentation")

        cluster_data = df[X_cols]

        kmeans = KMeans(n_clusters=3)

        clusters = kmeans.fit_predict(cluster_data)

        df["Segment"] = clusters

        fig2 = px.scatter(df, x=X_cols[0], y=X_cols[1], color="Segment")

        st.plotly_chart(fig2)

        # Radar chart
        st.subheader("Luxury Motivation Radar")

        radar = cluster_data.mean()

        radar_df = pd.DataFrame({
            "Driver":radar.index,
            "Value":radar.values
        })

        fig3 = px.line_polar(radar_df, r="Value", theta="Driver", line_close=True)

        st.plotly_chart(fig3)

        # Simulator
        st.subheader("Luxury Purchase Simulator")

        inputs = {}

        for col in X_cols:

            inputs[col] = st.slider(col,1,20,10)

        pred = model.predict([[1]+list(inputs.values())])

        st.metric("Predicted Purchase Score",round(pred[0],2))