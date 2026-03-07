import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
import networkx as nx
import numpy as np

st.set_page_config(page_title="HayaGriva Luxury Analytics", layout="wide")

# -------------------------
# PREMIUM PURPLE UI THEME
# -------------------------

st.markdown("""
<style>

.stApp {
background: linear-gradient(135deg,#0f0026,#2b0057,#5e17eb);
color:white;
}

h1,h2,h3,h4{
color:white;
}

.metric-box{
background:rgba(255,255,255,0.1);
padding:15px;
border-radius:10px;
}

.block-container{
padding-top:2rem;
}

</style>
""", unsafe_allow_html=True)

st.title("HayaGriva Luxury Consumer Intelligence Platform")

st.write("Advanced behavioral analytics for luxury purchase intention")

uploaded_file = st.file_uploader(
"Upload Luxury Consumer Dataset",
type=["xlsx","csv"]
)

# -------------------------
# MAIN ANALYSIS
# -------------------------

if uploaded_file:

    df = pd.read_excel(uploaded_file)

    emotion = df.iloc[:,5]
    celebrity = df.iloc[:,10]
    fomo = df.iloc[:,15]
    purchase = df.iloc[:,20]

    st.subheader("Dataset Overview")

    st.dataframe(df.head())

# -------------------------
# CPM CAUSAL MODEL
# -------------------------

    def regression(x,y):
        x = np.array(x)
        y = np.array(y)
        b = np.polyfit(x,y,1)
        return b[0]

    path_CF = regression(celebrity,fomo)
    path_FE = regression(fomo,emotion)
    path_EP = regression(emotion,purchase)
    path_CE = regression(celebrity,emotion)

    st.subheader("Causal Path Model")

    col1,col2,col3,col4 = st.columns(4)

    col1.metric("Celebrity → FOMO", round(path_CF,3))
    col2.metric("FOMO → Emotion", round(path_FE,3))
    col3.metric("Emotion → Purchase", round(path_EP,3))
    col4.metric("Celebrity → Emotion", round(path_CE,3))

# -------------------------
# HEATMAP
# -------------------------

    st.subheader("Psychological Influence Heatmap")

    corr = df.iloc[:,[5,10,15,20]].corr()

    fig_heat = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="purples"
    )

    st.plotly_chart(fig_heat,use_container_width=True)

# -------------------------
# CAUSAL NETWORK
# -------------------------

    st.subheader("Luxury Decision Network")

    G = nx.DiGraph()

    G.add_edges_from([
        ("Celebrity","FOMO"),
        ("FOMO","Emotion"),
        ("Emotion","Purchase"),
        ("Celebrity","Emotion")
    ])

    pos = nx.spring_layout(G)

    edge_x=[]
    edge_y=[]

    for edge in G.edges():
        x0,y0 = pos[edge[0]]
        x1,y1 = pos[edge[1]]
        edge_x.extend([x0,x1,None])
        edge_y.extend([y0,y1,None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode='lines',
        line=dict(width=2,color="white")
    )

    node_x=[]
    node_y=[]

    for node in G.nodes():
        x,y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        text=list(G.nodes()),
        textposition="bottom center",
        marker=dict(
            size=20,
            color="#c084fc"
        )
    )

    fig_net = go.Figure(data=[edge_trace,node_trace])

    fig_net.update_layout(
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)'
    )

    st.plotly_chart(fig_net,use_container_width=True)

# -------------------------
# CONSUMER SEGMENTATION
# -------------------------

    st.subheader("Luxury Consumer Segmentation")

    X = df[[df.columns[5],df.columns[10],df.columns[15]]]

    kmeans = KMeans(n_clusters=3)
    df["segment"] = kmeans.fit_predict(X)

    fig_seg = px.scatter(
        df,
        x=df.columns[5],
        y=df.columns[20],
        color="segment",
        color_continuous_scale="purples"
    )

    st.plotly_chart(fig_seg,use_container_width=True)

# -------------------------
# AI PERSONA GENERATOR
# -------------------------

    st.subheader("AI Luxury Consumer Personas")

    persona_summary = df.groupby("segment")[[df.columns[5],df.columns[10],df.columns[15],df.columns[20]]].mean()

    insights = []

    for i,row in persona_summary.iterrows():

        emotion_score = row[df.columns[5]]
        celeb_score = row[df.columns[10]]
        fomo_score = row[df.columns[15]]

        if emotion_score > celeb_score and emotion_score > fomo_score:

            persona="Emotion-Driven Buyer"
            desc="Purchases luxury primarily for emotional gratification."

        elif celeb_score > emotion_score:

            persona="Status-Seeking Consumer"
            desc="Strongly influenced by prestige and celebrity endorsement."

        else:

            persona="Social Comparison Buyer"
            desc="Driven by social pressure and FOMO."

        insights.append([i,persona,emotion_score,celeb_score,fomo_score])

        st.markdown(f"""
### Segment {i}: {persona}

Emotion Score: **{emotion_score:.2f}**

Celebrity Influence: **{celeb_score:.2f}**

FOMO Level: **{fomo_score:.2f}**

**Strategic Insight**

{desc}
""")

# -------------------------
# PURCHASE SIMULATOR
# -------------------------

    st.subheader("Luxury Purchase Prediction Simulator")

    colA,colB,colC = st.columns(3)

    emotion_input = colA.slider("Emotion",1,20,10)
    celebrity_input = colB.slider("Celebrity",1,20,10)
    fomo_input = colC.slider("FOMO",1,20,10)

    score = (
        0.63*emotion_input +
        0.16*celebrity_input +
        0.26*fomo_input
    )

    st.success(f"Predicted Purchase Intention Score: {round(score,2)}")

# -------------------------
# THEORETICAL ANALYSIS
# -------------------------

    st.subheader("Strategic Behavioral Insights")

    st.write(f"""
The causal path analysis reveals that **emotional response is the dominant driver of luxury purchase intention (β = {round(path_EP,3)})**.

Key strategic implications:

• Emotional engagement plays a stronger role than celebrity endorsement.

• Social comparison (FOMO) acts as a secondary trigger for luxury consumption.

• Luxury marketing strategies should emphasize **experiential storytelling and emotional resonance**.

These findings support the **Affective Decision-Making Theory in luxury consumer behaviour**.
""")

# -------------------------
# DOWNLOADABLE REPORT
# -------------------------

    st.subheader("Download Analytics Report")

    report = persona_summary.reset_index()

    csv = report.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="Download Luxury Consumer Intelligence Report",
        data=csv,
        file_name="hayagriva_consumer_report.csv",
        mime="text/csv"
    )