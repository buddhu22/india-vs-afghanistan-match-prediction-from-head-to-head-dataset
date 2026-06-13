import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import shap
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Cricket Match Prediction System",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for cricket match theme
st.markdown("""
    <style>
    :root {
        --field: #00E676;
        --field-dark: #00B85C;
        --saffron: #FF9933;
        --ink: #070A0F;
        --panel: rgba(12, 16, 24, 0.82);
        --panel-soft: rgba(18, 24, 34, 0.72);
        --line: rgba(255, 255, 255, 0.12);
        --muted: #B9C3D0;
    }
    .main { background: transparent; }
    .stApp {
        color: #FFFFFF;
        background:
            radial-gradient(circle at 18% 12%, rgba(255, 153, 51, 0.22), transparent 32rem),
            radial-gradient(circle at 86% 16%, rgba(0, 230, 118, 0.18), transparent 34rem),
            linear-gradient(180deg, rgba(3, 7, 13, 0.68), rgba(3, 7, 13, 0.92)),
            url("https://images.unsplash.com/photo-1540747913346-19e32dc3e97e?auto=format&fit=crop&w=1800&q=80");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    .block-container {
        max-width: 1180px;
        background: linear-gradient(180deg, var(--panel), rgba(8, 11, 17, 0.78));
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 2.3rem 2.6rem 3rem;
        margin-top: 1.4rem;
        margin-bottom: 2rem;
        box-shadow: 0 24px 70px rgba(0, 0, 0, 0.38);
        backdrop-filter: blur(6px);
    }
    [data-testid="stSidebar"] {
        background:
            linear-gradient(180deg, rgba(255, 153, 51, 0.22), rgba(255, 255, 255, 0.04), rgba(0, 230, 118, 0.16)),
            rgba(7, 10, 16, 0.94);
        border-right: 1px solid var(--line);
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] img {
        display: block;
        margin: 0 auto 0.5rem;
        filter: drop-shadow(0 10px 22px rgba(0, 0, 0, 0.36));
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        text-align: center;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label {
        background: rgba(255, 255, 255, 0.055);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 0.48rem 0.72rem;
        margin-bottom: 0.45rem;
        transition: all 160ms ease;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: rgba(0, 230, 118, 0.1);
        border-color: rgba(0, 230, 118, 0.35);
    }
    h1, h2, h3 {
        color: var(--field);
        letter-spacing: 0;
    }
    h1 {
        padding-bottom: 0.55rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }
    p, label, span, div {
        color: inherit;
    }
    .stMarkdown p {
        color: var(--muted);
        font-size: 1.02rem;
    }
    .stButton>button {
        width: 100%;
        min-height: 3rem;
        background: linear-gradient(135deg, var(--field), #9CFFB7);
        color: #06110B;
        border-radius: 8px;
        border: none;
        box-shadow: 0 12px 28px rgba(0, 230, 118, 0.22);
        font-weight: bold;
        transition: transform 150ms ease, box-shadow 150ms ease, filter 150ms ease;
    }
    .stButton>button:hover {
        color: #06110B;
        filter: brightness(0.96);
        transform: translateY(-1px);
        box-shadow: 0 16px 34px rgba(0, 230, 118, 0.28);
    }
    .stButton>button:active {
        transform: translateY(0);
    }
    .metric-card {
        min-height: 135px;
        background:
            linear-gradient(135deg, rgba(255, 255, 255, 0.07), rgba(255, 255, 255, 0.02)),
            var(--panel-soft);
        padding: 22px;
        border-radius: 8px;
        border: 1px solid var(--line);
        border-left: 5px solid var(--field);
        margin-bottom: 20px;
        box-shadow: 0 16px 34px rgba(0, 0, 0, 0.2);
    }
    .metric-card h2,
    .metric-card h3,
    .metric-card h4,
    .metric-card p {
        margin: 0.25rem 0;
    }
    .prediction-box {
        background:
            linear-gradient(135deg, rgba(0, 230, 118, 0.18), rgba(255, 153, 51, 0.12)),
            rgba(10, 15, 22, 0.88);
        padding: 34px;
        border-radius: 8px;
        text-align: center;
        border: 1px solid rgba(0, 230, 118, 0.42);
        box-shadow: 0 20px 48px rgba(0, 230, 118, 0.16);
    }
    .app-footer {
        margin-top: 2.5rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(255, 255, 255, 0.12);
        text-align: center;
        color: var(--muted);
        font-size: 0.95rem;
    }
    .app-footer strong {
        color: var(--field);
    }
    div[data-baseweb="select"] > div,
    div[data-baseweb="slider"] {
        background: rgba(255, 255, 255, 0.04);
        border-radius: 8px;
    }
    [data-testid="stSelectbox"],
    [data-testid="stSlider"] {
        background: rgba(255, 255, 255, 0.045);
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-radius: 8px;
        padding: 0.85rem 1rem 1rem;
    }
    [data-testid="stDataFrame"],
    [data-testid="stTable"],
    [data-testid="stImage"],
    [data-testid="stPlotlyChart"],
    [data-testid="stPyplot"] {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 14px 36px rgba(0, 0, 0, 0.22);
    }
    hr {
        border-color: rgba(255, 255, 255, 0.12);
        margin: 1.8rem 0;
    }
    @media (max-width: 768px) {
        .block-container {
            padding: 1.35rem 1rem 2rem;
            margin-top: 0.4rem;
            border-radius: 0;
        }
        .metric-card,
        .prediction-box {
            padding: 18px;
        }
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    try:
        best_winner_model = joblib.load('models/best_winner_model.pkl')
        scorer_model      = joblib.load('models/top_scorer_model.pkl')
        bowler_model      = joblib.load('models/top_bowler_model.pkl')
        encoders          = joblib.load('models/label_encoders.pkl')
        scaler            = joblib.load('models/scaler.pkl')
        features_list     = joblib.load('models/features_list.pkl')
        winner_metrics    = joblib.load('models/winner_model_metrics.pkl')
        shap_explainer    = joblib.load('models/shap_explainer.pkl')
        baseline_X        = joblib.load('models/shap_baseline_X.pkl')
        return best_winner_model, scorer_model, bowler_model, encoders, scaler, features_list, winner_metrics, shap_explainer, baseline_X
    except Exception as e:
        st.error(f"Error loading models: {e}. Please ensure the models are trained.")
        return None, None, None, None, None, None, None, None, None

@st.cache_data
def load_data():
    if os.path.exists('data/synthetic_match_data.csv'):
        return pd.read_csv('data/synthetic_match_data.csv')
    return pd.DataFrame()

models = load_models()
if models[0] is not None:
    best_winner_model, scorer_model, bowler_model, encoders, scaler, features_list, winner_metrics, shap_explainer, baseline_X = models
    df = load_data()

    # Sidebar Navigation
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3303/3303666.png", width=100)
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", [
        "🏏 Match Prediction",
        "👥 Team Comparison",
        "⭐ Player Prediction",
        "📊 Data Insights",
        "⚙️ Model Performance"
    ])

    # ─────────────────────────────────────────────────────────
    # PAGE 1: Match Prediction
    # ─────────────────────────────────────────────────────────
    if page == "🏏 Match Prediction":
        st.title("🏏 India vs Afghanistan: Match Prediction")
        st.markdown("Enter match conditions below to predict the outcome and key performers.")

        col1, col2 = st.columns(2)
        with col1:
            venue        = encoders['venue'].classes_[0]   # Default venue (hidden from UI)
            toss_winner  = st.selectbox("Toss Winner",  ['India', 'Afghanistan'])
            toss_decision= st.selectbox("Toss Decision",['Bat', 'Field'])

        with col2:
            ind_recent_form = st.slider("India Recent Form Index",        0.0, 1.0, 0.7)
            afg_recent_form = st.slider("Afghanistan Recent Form Index",   0.0, 1.0, 0.5)

            # Derived fields
            if venue in ['Delhi', 'Mumbai', 'Chennai']:
                venue_type = 'Home_India'
            elif venue == 'Kabul':
                venue_type = 'Home_Afg'
            else:
                venue_type = 'Neutral'

            if (toss_winner == 'India' and toss_decision == 'Bat') or \
               (toss_winner == 'Afghanistan' and toss_decision == 'Field'):
                batting_first = 'India'
            else:
                batting_first = 'Afghanistan'

        if st.button("🔮 Predict Match Outcome"):
            with st.spinner("Analyzing data and generating prediction..."):
                input_data = pd.DataFrame(columns=features_list)
                input_data.loc[0] = 0

                input_data['venue_encoded']       = encoders['venue'].transform([venue])[0]
                input_data['venue_type_encoded']  = encoders['venue_type'].transform([venue_type])[0]
                input_data['toss_winner_encoded'] = encoders['toss_winner'].transform([toss_winner])[0]
                input_data['toss_decision_encoded']= encoders['toss_decision'].transform([toss_decision])[0]
                input_data['batting_first_encoded']= encoders['batting_first'].transform([batting_first])[0]

                input_data['ind_recent_form']     = ind_recent_form
                input_data['afg_recent_form']     = afg_recent_form
                input_data['ind_batting_strength']= 85
                input_data['ind_bowling_strength']= 88
                input_data['afg_batting_strength']= 75
                input_data['afg_bowling_strength']= 80
                input_data['ind_player_form_idx'] = 80
                input_data['afg_player_form_idx'] = 70
                input_data['ind_h2h_win_pct']     = 0.82
                input_data['ind_avg_score']       = 290
                input_data['afg_avg_score']       = 245

                input_scaled = scaler.transform(input_data)

                prob       = best_winner_model.predict_proba(input_scaled)[0]
                pred_class = best_winner_model.predict(input_scaled)[0]

                winner   = 'India' if pred_class == 1 else 'Afghanistan'
                win_prob = prob[1] if pred_class == 1 else prob[0]

                scorer_pred = scorer_model.predict(input_scaled)[0]
                bowler_pred = bowler_model.predict(input_scaled)[0]
                top_scorer  = encoders['top_scorer'].inverse_transform([scorer_pred])[0]
                top_bowler  = encoders['top_bowler'].inverse_transform([bowler_pred])[0]

                if winner == 'India':
                    expected_score = int(280 + (win_prob - 0.5) * 100)
                else:
                    expected_score = int(250 + (win_prob - 0.5) * 80)

                st.markdown("---")

                st.markdown(f"""
                <div class="prediction-box">
                    <h2 style="color: #00E676; margin-bottom: 5px;">🏆 Predicted Winner: {winner}</h2>
                    <h3 style="color: #ffffff; font-weight: 300;">Winning Probability: {win_prob*100:.1f}%</h3>
                    <p style="color: #aaaaaa;">Confidence Score: {win_prob*100:.1f} / 100</p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>⭐ Top Batsman</h4>
                        <h3>{top_scorer}</h3>
                    </div>""", unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>🎯 Top Bowler</h4>
                        <h3>{top_bowler}</h3>
                    </div>""", unsafe_allow_html=True)
                with c3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>📈 Expected Team Score</h4>
                        <h3>~{expected_score} Runs</h3>
                    </div>""", unsafe_allow_html=True)

                st.subheader("🧠 Match Outcome Explanation (SHAP)")
                st.markdown("The chart below shows which features contributed most to this specific prediction.")
                try:
                    shap_values = shap_explainer(input_scaled)
                    fig, ax = plt.subplots(figsize=(10, 4))
                    plt.style.use('dark_background')
                    shap.waterfall_plot(shap_values[0], show=False)
                    st.pyplot(fig)
                except Exception:
                    st.warning("SHAP explanation not fully compatible with the selected model.")

    # ─────────────────────────────────────────────────────────
    # PAGE 2: Team Comparison
    # ─────────────────────────────────────────────────────────
    elif page == "👥 Team Comparison":
        st.title("👥 Team Comparison: India vs Afghanistan")
        if not df.empty:
            col1, col2 = st.columns(2)
            ind_wins = len(df[df['winner'] == 'India'])
            afg_wins = len(df[df['winner'] == 'Afghanistan'])

            with col1:
                st.markdown(f"""
                <div class="metric-card" style="text-align: center;">
                    <h2>India</h2>
                    <h3>{ind_wins} Wins</h3>
                    <p>Avg Score: {df['ind_score'].mean():.1f}</p>
                </div>""", unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-card" style="text-align: center;">
                    <h2>Afghanistan</h2>
                    <h3>{afg_wins} Wins</h3>
                    <p>Avg Score: {df['afg_score'].mean():.1f}</p>
                </div>""", unsafe_allow_html=True)

            st.subheader("Score Distribution Comparison")
            fig, ax = plt.subplots(figsize=(10, 5))
            plt.style.use('dark_background')
            sns.kdeplot(df['ind_score'], fill=True, label='India',       color='#00E676')
            sns.kdeplot(df['afg_score'], fill=True, label='Afghanistan', color='#ff3333')
            plt.legend()
            st.pyplot(fig)

    # ─────────────────────────────────────────────────────────
    # PAGE 3: Player Prediction
    # ─────────────────────────────────────────────────────────
    elif page == "⭐ Player Prediction":
        st.title("⭐ Player Performance Insights")
        if not df.empty:
            st.markdown("### Top Performers in History")
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Most Times Top Scorer")
                scorer_counts = df['top_scorer'].value_counts().head(5)
                st.bar_chart(scorer_counts)

            with col2:
                st.subheader("Most Times Top Bowler")
                bowler_counts = df['top_bowler'].value_counts().head(5)
                st.bar_chart(bowler_counts)

            st.markdown("### Player Form Index Correlation with Wins")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(data=df.sample(200), x='ind_player_form_idx', y='ind_score',
                            hue='winner', palette={'India':'#00E676', 'Afghanistan':'#ff3333'})
            st.pyplot(fig)

    # ─────────────────────────────────────────────────────────
    # PAGE 4: Data Insights
    # ─────────────────────────────────────────────────────────
    elif page == "📊 Data Insights":
        st.title("📊 Exploratory Data Analysis")
        if not df.empty:
            st.subheader("Toss Decision Impact on Match Outcome")
            toss_impact = pd.crosstab(df['toss_decision'], df['winner'], normalize='index') * 100
            st.bar_chart(toss_impact)

            st.subheader("Batting First vs Chasing")
            batting_first_wins = df[df['batting_first'] == df['winner']].shape[0] / len(df) * 100
            chasing_wins = 100 - batting_first_wins

            fig, ax = plt.subplots()
            ax.pie([batting_first_wins, chasing_wins],
                   labels=['Batting First Wins', 'Chasing Wins'],
                   autopct='%1.1f%%',
                   colors=['#00E676', '#1E2127'],
                   textprops={'color': 'w'})
            st.pyplot(fig)

            st.subheader("Correlation Heatmap of Key Features")
            num_cols = ['ind_score', 'afg_score', 'ind_recent_form', 'afg_recent_form',
                        'ind_batting_strength', 'afg_batting_strength']
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(df[num_cols].corr(), annot=True, cmap='viridis')
            st.pyplot(fig)

    # ─────────────────────────────────────────────────────────
    # PAGE 5: Model Performance
    # ─────────────────────────────────────────────────────────
    elif page == "⚙️ Model Performance":
        st.title("⚙️ Model Performance & Explainability")

        st.subheader("Algorithm Comparison")
        metrics_df = pd.DataFrame(winner_metrics).T
        st.dataframe(metrics_df.style.background_gradient(cmap='viridis'))

        st.markdown(f"**Selected Best Model**: `{best_winner_model.__class__.__name__}`")

        st.subheader("Global Feature Importance (SHAP)")
        st.markdown("Which features generally impact the prediction the most across all matches?")
        try:
            shap_values = shap_explainer(baseline_X)
            fig, ax = plt.subplots(figsize=(10, 6))
            plt.style.use('dark_background')
            shap.summary_plot(shap_values, baseline_X, show=False)
            st.pyplot(fig)
        except Exception:
            st.warning("SHAP summary plot not available for this model type.")

else:
    st.error("No models found. Please train the models by running the training pipeline.")

st.markdown("""
    <div class="app-footer">
        Made by <strong>Abhay Mishra</strong>
    </div>
""", unsafe_allow_html=True)
