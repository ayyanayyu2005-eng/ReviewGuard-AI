
import streamlit as st
import pickle
import pandas as pd
import plotly.express as px

# ---------------- PAGE SETTINGS ----------------

st.set_page_config(
    page_title="ReviewGuard AI",
    page_icon="🛡️",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------

with open("fake_review_model.pkl", "rb") as f:
    model = pickle.load(f)

df = pd.read_csv("reviews_dataset.csv")

if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- CUSTOM DESIGN ----------------

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #f0f4ff, #f8f5ff);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #171c45, #31256b);
}

[data-testid="stSidebar"] * {
    color: white !important;
}

.main-title {
    background: linear-gradient(90deg, #4435c7, #a13bd4, #168fc9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 38px;
    font-weight: 800;
}

.subtitle {
    color: #68738b;
    font-size: 17px;
}

.metric-card {
    background: white;
    padding: 23px;
    border-radius: 18px;
    box-shadow: 0px 5px 20px rgba(50, 50, 120, 0.08);
    border: 1px solid #e8eaff;
}

.metric-label {
    color: #78829b;
    font-size: 14px;
    font-weight: 600;
}

.metric-value {
    color: #29245f;
    font-size: 30px;
    font-weight: 800;
}

.section-card {
    background: white;
    padding: 25px;
    border-radius: 18px;
    border: 1px solid #e6e8f5;
    box-shadow: 0px 5px 18px rgba(50, 50, 120, 0.06);
}

div.stButton > button {
    background: linear-gradient(90deg, #5145cd, #8d43ce);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 12px 25px;
    font-weight: 700;
}

div.stButton > button:hover {
    background: linear-gradient(90deg, #3024a5, #7129ad);
    color: white;
    border: none;
}

.stTextArea textarea {
    border-radius: 14px;
    border: 2px solid #d9d7f5;
    background: white;
}

h1, h2, h3 {
    color: #29245f;
}

</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------

st.sidebar.markdown("# 🛡️ ReviewGuard")
st.sidebar.markdown("### AI Review Intelligence")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "🏠 Dashboard",
        "🔍 Review Analyzer",
        "📚 All Reviews",
        "📋 History"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Powered by Machine Learning")
st.sidebar.caption("TF-IDF + Logistic Regression")

# ---------------- DATA ----------------

total = len(df)
fake = int((df["label"] == "Fake").sum())
genuine = int((df["label"] == "Genuine").sum())

# ---------------- DASHBOARD ----------------

if page == "🏠 Dashboard":

    st.markdown(
        '<div class="main-title">🛡️ ReviewGuard AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Intelligent Review Detection & Analysis Platform</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)

    metrics = [
        (c1, "TOTAL REVIEWS", f"{total:,}", "📊"),
        (c2, "FAKE REVIEWS", f"{fake:,}", "🚨"),
        (c3, "GENUINE REVIEWS", f"{genuine:,}", "✅"),
        (c4, "MODEL ACCURACY", "86.88%", "🎯")
    ]

    for col, label, value, emoji in metrics:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{emoji} {label}</div>
                <div class="metric-value">{value}</div>
            </div>
            """, unsafe_allow_html=True)

    st.write("")

    left, right = st.columns(2)

    with left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("📊 Review Distribution")

        fig = px.pie(
            df,
            names="label",
            hole=0.58,
            color="label",
            color_discrete_map={
                "Fake": "#F05270",
                "Genuine": "#16B89A"
            }
        )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=10, b=10, l=10, r=10)
        )

        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("📈 Review Comparison")

        counts = df["label"].value_counts().reset_index()
        counts.columns = ["Review Type", "Count"]

        fig2 = px.bar(
            counts,
            x="Review Type",
            y="Count",
            color="Review Type",
            color_discrete_map={
                "Fake": "#F05270",
                "Genuine": "#16B89A"
            },
            text="Count"
        )

        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False
        )

        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="section-card">', unsafe_allow_html=True)

    st.subheader("✨ About ReviewGuard AI")

    st.write("""
    ReviewGuard AI is a machine-learning prototype that analyses
    written reviews and classifies them as suspicious-looking or
    genuine-looking.

    **Technologies:** Python | Pandas | Scikit-learn |
    TF-IDF | Logistic Regression | Streamlit | Plotly
    """)

    st.markdown('</div>', unsafe_allow_html=True)

    st.warning(
        "Academic prototype trained on hotel reviews. "
        "Predictions on product reviews are experimental."
    )

# ---------------- REVIEW ANALYZER ----------------

elif page == "🔍 Review Analyzer":

    st.markdown(
        '<div class="main-title">🔍 AI Review Analyzer</div>',
        unsafe_allow_html=True
    )

    st.write("Check your review using our Machine Learning model.")

    st.markdown('<div class="section-card">', unsafe_allow_html=True)

    review = st.text_area(
        "Enter your review",
        height=180,
        placeholder="Write or paste your review here..."
    )

    if st.button("🚀 Analyze Review", type="primary"):

        if review.strip():

            prediction = model.predict([review])[0]
            probabilities = model.predict_proba([review])[0]
            confidence = max(probabilities) * 100

            st.markdown("---")
            st.subheader("Analysis Result")

            if prediction == "Fake":
                st.error("🚨 Suspicious-looking Review")
            else:
                st.success("✅ Genuine-looking Review")

            st.metric(
                "Prediction Confidence",
                f"{confidence:.2f}%"
            )

            st.progress(float(confidence / 100))

            st.session_state.history.append({
                "Review": review,
                "Prediction": prediction,
                "Confidence": f"{confidence:.2f}%"
            })

            st.caption(
                "Confidence is the model's estimate, not proof of authenticity."
            )

        else:
            st.warning("Please enter a review first.")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- ALL REVIEWS ----------------

elif page == "📚 All Reviews":

    st.markdown(
        '<div class="main-title">📚 Reviews Explorer</div>',
        unsafe_allow_html=True
    )

    st.write(f"Explore your dataset of {total:,} labelled reviews.")

    search = st.text_input("🔎 Search review")

    selected = st.selectbox(
        "Filter reviews",
        ["All", "Fake", "Genuine"]
    )

    filtered = df.copy()

    if selected != "All":
        filtered = filtered[filtered["label"] == selected]

    if search.strip():
        filtered = filtered[
            filtered["review"].str.contains(
                search,
                case=False,
                na=False
            )
        ]

    st.info(f"Showing {len(filtered):,} reviews")

    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "⬇️ Download Reviews",
        filtered.to_csv(index=False).encode("utf-8"),
        "reviews.csv",
        "text/csv"
    )

# ---------------- HISTORY ----------------

elif page == "📋 History":

    st.markdown(
        '<div class="main-title">📋 Prediction History</div>',
        unsafe_allow_html=True
    )

    if st.session_state.history:

        history_df = pd.DataFrame(st.session_state.history)

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        counts = history_df["Prediction"].value_counts()

        st.subheader("📊 Your Prediction Statistics")
        st.bar_chart(counts)

        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()

    else:
        st.info("No predictions yet. Visit Review Analyzer to start.")

# ---------------- FOOTER ----------------

st.markdown("---")

st.markdown(
    "<center>🛡️ <b>ReviewGuard AI</b> | Built with Python & Machine Learning</center>",
    unsafe_allow_html=True
)
