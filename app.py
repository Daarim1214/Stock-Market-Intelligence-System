
import os
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import shap
import matplotlib.pyplot as plt
import cohere
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_text_splitters import RecursiveCharacterTextSplitter

COHERE_API_KEY = "YOUR_COHERE_API_KEY"

# ======================================================
# LOAD EMBEDDING MODEL
# ======================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

# ======================================================
# LOAD NEWS FILES
# ======================================================

@st.cache_data
def load_news():

    assets = [
        "^BSESN",
        "RELIANCE.NS",
        "TCS.NS",
        "INFY.NS",
        "HDFCBANK.NS"
    ]

    news_data = {}

    for ticker in assets:

       filename = f"news/news_{ticker}.csv"

    if os.path.exists(filename):

          df = pd.read_csv(filename)

          news_data[ticker] = df

          return news_data


news_data = load_news()

# ======================================================
# CLEAN NEWS DATA
# ======================================================

for ticker, df in news_data.items():

    if "publishedAt" in df.columns:
        df["publishedAt"] = pd.to_datetime(
            df["publishedAt"],
            errors="coerce"
        )

    df.drop_duplicates(inplace=True)

    for col in ["title", "description", "content"]:

        if col in df.columns:
            df[col] = df[col].fillna("")

    news_data[ticker] = df

# ======================================================
# CREATE DOCUMENTS
# ======================================================

news_documents = {}

for ticker, df in news_data.items():

    document = ""

    for _, row in df.iterrows():

        article = f"""
Title: {row['title']}

Description:
{row['description']}

Content:
{row['content']}

"""

        document += article + "\n"

    news_documents[ticker] = document

# ======================================================
# CREATE CHUNKS
# ======================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

news_chunks = {}

for ticker, text in news_documents.items():

    chunks = text_splitter.split_text(text)

    news_chunks[ticker] = chunks

# ======================================================
# CREATE EMBEDDINGS
# ======================================================

news_embeddings = {}

for ticker, chunks in news_chunks.items():

    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True
    )

    news_embeddings[ticker] = embeddings


# ======================================================
# RETRIEVER
# ======================================================

def retrieve_news(query, ticker, top_k=3):

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True
    )

    similarities = cosine_similarity(
        query_embedding,
        news_embeddings[ticker]
    )[0]

    top_indices = similarities.argsort()[-top_k:][::-1]

    retrieved_chunks = [
        news_chunks[ticker][i]
        for i in top_indices
    ]

    return retrieved_chunks

# ======================================================
# RAG FUNCTION
# ======================================================

def ask_rag(question, ticker, api_key):

    retrieved_chunks = retrieve_news(
        question,
        ticker
    )

    context = "\n\n".join(retrieved_chunks)

    co = cohere.ClientV2(api_key)

    prompt = f"""
Use ONLY the context below to answer the question.

Context:
{context}

Question:
{question}
"""

    response = co.chat(

        model="command-a-03-2025",

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.message.content[0].text


# ======================================================
# PAGE CONFIG
# ======================================================

st.set_page_config(
    page_title="Stock Market Intelligence System",
    page_icon="📈",
    layout="wide"
)

# ======================================================
# TITLE
# ======================================================

st.title("📈 Stock Market Intelligence System")

st.write("Capstone Project")

# ======================================================
# SIDEBAR
# ======================================================

st.sidebar.title("Asset Selection")

assets = [
    "^BSESN",
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS"
]

selected_asset = st.sidebar.selectbox(
    "Choose an Asset",
    assets
)

st.sidebar.success(f"Selected: {selected_asset}")


# ======================================================
# TABS
# ======================================================

tab1, tab2, tab3 = st.tabs(
    [
        "📈 Predictions",
        "💬 Chat",
        "📊 Comparison"
    ]
)

# ======================================================
# TAB 1 : Prediction
# ======================================================

with tab1:

    st.header("Predictions")

    st.subheader(f"Selected Asset : {selected_asset}")

    # Load stock data
    file_name = f"data/data_{selected_asset}.csv"

    df = pd.read_csv(file_name)

    # Convert Date column
    df["Date"] = pd.to_datetime(df["Date"])

    # Price Chart
    fig = px.line(
        df,
        x="Date",
        y="Close",
        title=f"{selected_asset} Closing Price"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------
    # Feature Engineering
    # ---------------------------------------

    df["ma_7"] = df["Close"].rolling(7).mean()
    df["ma_30"] = df["Close"].rolling(30).mean()

    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    df = df.dropna()

    feature_columns = ["Close","Volume","ma_7","ma_30"]

    X = df[feature_columns]
    y = df["target"]

    # ---------------------------------------
    # Scale Data
    # ---------------------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # ---------------------------------------
    # Train Model
    # ---------------------------------------

    model = RandomForestClassifier(n_estimators=100, random_state=42)

    model.fit(X_scaled, y)

    # ---------------------------------------
    # Predict Latest Day
    # ---------------------------------------

    latest = X_scaled[-1].reshape(1, -1)

    prediction = model.predict(latest)[0]

    confidence = model.predict_proba(latest)[0].max()

    # ---------------------------------------
    # SHAP Explainer
    # ---------------------------------------

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_scaled)

    st.subheader("Prediction")

    if prediction == 1:
        st.success("📈 Prediction : UP")
    else:
        st.error("📉 Prediction : DOWN")

    st.metric(
        "Confidence Score",
        f"{confidence*100:.2f}%"
    )

    # ---------------------------------------
    # SHAP Feature Importance
    # ---------------------------------------

    st.subheader("SHAP Feature Importance")

    fig = plt.figure(figsize=(10,6))

    shap.summary_plot(
        shap_values[:, :, 1],
        X_scaled,
        feature_names=feature_columns,
        plot_type="bar",
        show=False
    )

    st.pyplot(fig)
    plt.close(fig)

    # ---------------------------------------
    # SHAP Waterfall Plot
    # ---------------------------------------

    st.subheader("SHAP Waterfall Plot")

    sample = -1

    # SHAP values for Class 1 (UP)
    sample_shap = shap_values[sample][:, 1]

    explanation = shap.Explanation(
        values=sample_shap,
        base_values=explainer.expected_value[1],
        data=X_scaled[sample],
        feature_names=feature_columns
    )

    fig = plt.figure(figsize=(8,6))

    shap.plots.waterfall(
        explanation,
        show=False
    )

    st.pyplot(fig)
    plt.close(fig)



# ==========================================================
# TAB 2 : CHAT
# ==========================================================

with tab2:

    st.header("💬 Stock Market Chatbot")

    st.write("Ask questions related to the selected stock.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_question = st.chat_input("Ask your question...")

    if user_question:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_question
            }
        )

        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                answer = ask_rag(
                    user_question,
                    selected_asset,
                    COHERE_API_KEY
                )

                st.markdown(answer)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )


# ======================================================
# TAB 3 : Comparision
# ======================================================

with tab3:

    st.header("📊 Asset Comparison")

    comparison_data = []

    for asset in assets:

        # -------------------------
        # Load data
        # -------------------------
        df = pd.read_csv(f"data_{asset}.csv")

        df["ma_7"] = df["Close"].rolling(7).mean()
        df["ma_30"] = df["Close"].rolling(30).mean()

        df["target"] = (
            df["Close"].shift(-1) > df["Close"]
        ).astype(int)

        df = df.dropna()

        feature_columns = [
            "Close",
            "Volume",
            "ma_7",
            "ma_30"
        ]

        X = df[feature_columns]
        y = df["target"]

        # -------------------------
        # Scale
        # -------------------------
        scaler = StandardScaler()

        X_scaled = scaler.fit_transform(X)

        # -------------------------
        # Train Model
        # -------------------------
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

        model.fit(X_scaled, y)

        # -------------------------
        # Latest Prediction
        # -------------------------
        latest = X_scaled[-1].reshape(1, -1)

        prediction = model.predict(latest)[0]

        confidence = model.predict_proba(latest)[0].max()

        prediction_text = "UP" if prediction == 1 else "DOWN"

        comparison_data.append({

            "Asset": asset,
            "Close": round(df["Close"].iloc[-1], 2),
            "Volume": int(df["Volume"].iloc[-1]),
            "ma_7": round(df["ma_7"].iloc[-1], 2),
            "ma_30": round(df["ma_30"].iloc[-1], 2),
            "Prediction": prediction_text,
            "Confidence": round(confidence * 100, 2)

        })

    comparison_df = pd.DataFrame(comparison_data)

    # ======================================================
    # COMPARISON TABLE
    # ======================================================

    st.subheader("Comparison Table")

    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True
    )

    # ======================================================
    # BAR CHART OF METRICS
    # ======================================================

    st.subheader("Bar Chart of Metrics")

    metric = st.selectbox(
        "Select Metric",
        [
            "Close",
            "Volume",
            "ma_7",
            "ma_30",
            "Confidence"
        ]
    )

    fig = px.bar(
        comparison_df,
        x="Asset",
        y=metric,
        color="Asset",
        title=f"{metric} Comparison"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ======================================================
    # PREDICTION COMPARISON
    # ======================================================

    st.subheader("Prediction Comparison")

    prediction_colors = []

    for pred in comparison_df["Prediction"]:

        if pred == "UP":
            prediction_colors.append("🟢 UP")
        else:
            prediction_colors.append("🔴 DOWN")

    prediction_display = comparison_df[["Asset"]].copy()

    prediction_display["Prediction"] = prediction_colors

    st.dataframe(
        prediction_display,
        use_container_width=True,
        hide_index=True
    )
