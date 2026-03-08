import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.express as px

st.set_page_config(page_title="HayaGriva Luxury Consumer Intelligence", layout="wide")

st.title("HayaGriva Luxury Consumer Intelligence")

st.markdown("> **Luxury is the balance of design, beauty and highest quality. — Domenico De Sole**")

st.subheader("Effect of Emotions on Purchase Intention of Luxury Products")

# Upload dataset
st.sidebar.header("Upload Dataset")
emotion_file = st.sidebar.file_uploader("Upload Emotional Dataset", type=["xlsx"])

if emotion_file:

    df = pd.read_excel(emotion_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    st.subheader("Select Variables")

    X_cols = st.multiselect(
        "Select Independent Variables",
        numeric_cols
    )

    y_col = st.selectbox(
        "Select Dependent Variable",
        numeric_cols
    )

    if len(X_cols) > 0 and y_col:

        X = df[X_cols]
        y = df[y_col]

        model = LinearRegression()
        model.fit(X, y)

        coef = pd.DataFrame({
            "Driver": X_cols,
            "Impact": model.coef_
        })

        st.subheader("Psychological Driver Impact")

        fig = px.bar(
            coef,
            x="Driver",
            y="Impact",
            title="Impact of Psychological Drivers on Purchase Intention"
        )

        st.plotly_chart(fig)

        st.subheader("Correlation Matrix")

        corr = df.corr()

        fig2 = px.imshow(corr, text_auto=True)

        st.plotly_chart(fig2)

        st.subheader("Luxury Purchase Simulator")

        inputs = {}

        for col in X_cols:
            inputs[col] = st.slider(col, 1, 20, 10)

        pred = model.predict(pd.DataFrame([inputs]))

        st.metric("Predicted Purchase Score", round(pred[0],2))

        st.subheader("Insights")

        st.markdown("""
• Emotional response strongly influences luxury purchase behaviour  
• Celebrity endorsements increase aspirational perception  
• FOMO creates urgency in luxury buying decisions  
• Psychological drivers significantly explain luxury consumption behaviour
""")

st.subheader("Project Conclusion")

st.markdown("""
Emotion is the strongest driver of luxury purchase intention.

Luxury brands must focus on emotional storytelling, aspirational branding and exclusivity strategies to strengthen consumer engagement.
""")