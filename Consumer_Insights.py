import streamlit as st
import statsmodels.api as sm
import pandas as pd

st.title("Multiple Linear Regression")

df = st.session_state.get("data")

if df is None:
    st.warning("Upload dataset first")

else:

    cols = df.columns

    X = df[cols[:-1]]

    y = df[cols[-1]]

    X = sm.add_constant(X)

    model = sm.OLS(y,X).fit()

    st.subheader("Model Summary")

    st.text(model.summary())

    coef_table = pd.DataFrame({

        "Variable": model.params.index,
        "Coefficient": model.params.values,
        "Std Error": model.bse.values,
        "p-value": model.pvalues.values

    })

    st.subheader("SPSS Style Coefficient Table")

    st.dataframe(coef_table)

    st.metric("R²", round(model.rsquared,3))