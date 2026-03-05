import streamlit as st
import pandas as pd
import pickle

# ── Load model and encoder ───────────────────────────────────
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# ── Teams list ───────────────────────────────────────────────
teams = sorted([
    'Chennai Super Kings', 'Mumbai Indians', 'Kolkata Knight Riders',
    'Royal Challengers Bangalore', 'Sunrisers Hyderabad', 'Delhi Capitals',
    'Rajasthan Royals', 'Punjab Kings', 'Lucknow Super Giants', 'Gujarat Titans'
])

# ── Page config ──────────────────────────────────────────────
st.set_page_config(page_title="IPL Match Predictor", page_icon="🏏", layout="centered")

st.markdown("""
    <h1 style='text-align:center; color:#FF6B00;'>🏏 IPL Match Winner Predictor</h1>
    <p style='text-align:center; color:gray;'>Built with Real IPL Data (2008-2022) | Random Forest ML Model</p>
    <hr>
""", unsafe_allow_html=True)

# ── Inputs ───────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    team1 = st.selectbox("🏏 Batting First (Team 1)", teams)
with col2:
    team2 = st.selectbox("🎯 Batting Second (Team 2)", [t for t in teams if t != team1])

st.markdown("### First Innings Result")
col3, col4 = st.columns(2)
with col3:
    score1 = st.number_input("Score posted by Team 1", min_value=50, max_value=300, value=165)
with col4:
    wickets1 = st.number_input("Wickets lost by Team 1", min_value=0, max_value=10, value=6)

# ── Historical win rates ──────────────────────────────────────
win_rates = {
    'Chennai Super Kings': 0.59, 'Mumbai Indians': 0.57,
    'Kolkata Knight Riders': 0.52, 'Royal Challengers Bangalore': 0.47,
    'Sunrisers Hyderabad': 0.51, 'Delhi Capitals': 0.48,
    'Rajasthan Royals': 0.49, 'Punjab Kings': 0.44,
    'Lucknow Super Giants': 0.55, 'Gujarat Titans': 0.58
}

# ── Predict ───────────────────────────────────────────────────
if st.button("🔮 Predict Winner", use_container_width=True):
    try:
        t1_enc = le.transform([team1])[0]
        t2_enc = le.transform([team2])[0]
        run_rate1 = score1 / 20
        t1_wr = win_rates.get(team1, 0.5)
        t2_wr = win_rates.get(team2, 0.5)

        features = pd.DataFrame([[t1_enc, t2_enc, score1, wickets1, run_rate1, t1_wr, t2_wr]],
                                 columns=['team1_enc', 'team2_enc', 'score1', 'wickets1',
                                          'run_rate1', 'team1_winrate', 'team2_winrate'])

        prob = model.predict_proba(features)[0]
        team1_prob = prob[1] * 100
        team2_prob = prob[0] * 100
        winner = team1 if team1_prob > team2_prob else team2

        st.markdown("---")
        st.markdown(f"<h2 style='text-align:center; color:#00FF88;'>🏆 Predicted Winner: {winner}</h2>", unsafe_allow_html=True)

        col5, col6 = st.columns(2)
        with col5:
            st.metric(f"{team1} Win Probability", f"{team1_prob:.1f}%")
            st.progress(int(team1_prob))
        with col6:
            st.metric(f"{team2} Win Probability", f"{team2_prob:.1f}%")
            st.progress(int(team2_prob))

        st.info(f"Model trained on 1092 IPL matches (2008-2022) | Accuracy: 67.6%")

    except Exception as e:
        st.error(f"Error: {e}. Make sure both teams are different.")

st.markdown("---")
st.caption("Built by [Your Name] | Model: Random Forest | Data: Kaggle IPL Dataset")