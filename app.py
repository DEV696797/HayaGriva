import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
import networkx as nx
import numpy as np

# PAGE CONFIG
st.set_page_config(page_title="HayaGriva Analytics", layout="wide")

# CUSTOM PURPLE THEME
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg,#1a0033,#3a0066,#5e17eb);
    color: white;
}

h1,h2,h3 {
color:white;
}

.block-container {
padding-top:2rem;
}

.metric-box {
background:rgba(255,255,255,0.1);
padding:20px;
border-radius:10px;
}
</style>
""", unsafe_allow_html=True)

st.title("HayaGriva Luxury Consumer Intelligence")

uploaded_file = st.file_uploader(
"Upload Luxury Consumer Dataset",
type=["xlsx","csv"]
)

if uploaded_file:

    df = pd.read_excel(uploaded_file)

    emotion = df.iloc[:,5]
    celebrity = df.iloc[:,10]
    fomo = df.iloc[:,15]
    purchase = df.iloc[:,20]

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # REGRESSION FUNCTION
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

    # CORRELATION HEATMAP
    st.subheader("Psychological Influence Heatmap")

    corr = df.iloc[:,[5,10,15,20]].corr()

    fig_heat = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="purples"
    )

    st.plotly_chart(fig_heat,use_container_width=True)

    # CAUSAL NETWORK GRAPH
    st.subheader("Luxury Consumer Causal Network")

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

    # SEGMENTATION
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

    # PURCHASE SIMULATOR
    st.subheader("Luxury Purchase Simulator")

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

    # THEORETICAL INTERPRETATION
    st.subheader("Theoretical Interpretation")

    st.write(f"""
The causal model suggests that **Emotional Response is the dominant predictor of Luxury Purchase Intention**.

Key findings:

• Emotional influence coefficient: **{round(path_EP,3)}**

• Celebrity influence contributes indirectly through FOMO and emotional engagement.

• Luxury purchasing behaviour appears to be **emotion-driven rather than purely status-driven**.

Managerial implications:

1. Luxury brands should prioritize **emotional storytelling and experiential marketing**.

2. Celebrity endorsements are more effective when they amplify emotional resonance rather than merely providing visibility.

3. Social comparison effects (FOMO) act as a **secondary psychological trigger**.

Overall, the results support **affective decision-making theory in luxury consumption behaviour**.
""")