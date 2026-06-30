import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Page config
# ---------------------------------------------------------
st.set_page_config(
    page_title="IPL Win Probability Model",
    
    layout="wide"
)

# ---------------------------------------------------------
# Load model & supporting data (cached so it only loads once)
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    with open('ipl_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

@st.cache_data
def load_supporting_data():
    batter_lookup = pd.read_csv('batter_skill_lookup.csv')
    with open('teams_venues.json') as f:
        tv = json.load(f)
    choke_stats = pd.read_csv('choke_stats.csv')
    return batter_lookup, tv['teams'], tv['venues'], choke_stats

model = load_model()
batter_lookup, teams_list, venues_list, choke_stats = load_supporting_data()
median_strike_rate = batter_lookup['career_strike_rate'].median()

# ---------------------------------------------------------
# Prediction function
# ---------------------------------------------------------
def predict_win_prob(runs_required, balls_remaining, wickets_remaining,
                     runs_scored, balls_bowled, batter_skill):
    overs_completed = balls_bowled / 6
    crr = runs_scored / max(overs_completed, 0.1)
    rrr = (runs_required * 6) / max(balls_remaining, 1)
    pressure_index = rrr - crr

    X_input = pd.DataFrame([{
        'runs_required': runs_required,
        'balls_remaining': balls_remaining,
        'wickets_remaining': wickets_remaining,
        'crr': crr,
        'rrr': rrr,
        'pressure_index': pressure_index,
        'batter_skill': batter_skill
    }])

    prob = model.predict_proba(X_input)[0][1]
    return prob, crr, rrr, pressure_index

# ---------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------
st.sidebar.title(" IPL Win Probability")
page = st.sidebar.radio("Navigate", ["Live Predictor", "Choking Analysis", "About the Model"])

# ===========================================================
# PAGE 1 — Live Predictor
# ===========================================================
if page == "Live Predictor":
    st.title(" IPL Win Probability Calculator")
    st.markdown("Enter the current match situation to see live win probability for the chasing team.")

    col1, col2 = st.columns(2)

    with col1:
        batting_team = st.selectbox("Batting Team (Chasing)", teams_list, index=0)
        bowling_team = st.selectbox("Bowling Team (Defending)", teams_list, index=1)
        venue = st.selectbox("Venue", venues_list)

    with col2:
        target = st.number_input("Target Score", min_value=1, max_value=300, value=180)
        runs_scored = st.number_input("Current Score", min_value=0, max_value=300, value=95)
        overs_done = st.number_input("Overs Completed", min_value=0.0, max_value=20.0, value=12.0, step=0.1)
        wickets_fallen = st.number_input("Wickets Fallen", min_value=0, max_value=10, value=4)

    batter_name = st.selectbox(
        "Current Batter (optional — for skill-adjusted prediction)",
        ["Use Average Player"] + sorted(batter_lookup['batter'].unique().tolist())
    )

    if batter_name == "Use Average Player":
        batter_skill = median_strike_rate
    else:
        batter_skill = batter_lookup.loc[
            batter_lookup['batter'] == batter_name, 'career_strike_rate'
        ].values[0]

    balls_bowled = int(overs_done * 6)
    runs_required = target - runs_scored
    balls_remaining = 120 - balls_bowled
    wickets_remaining = 10 - wickets_fallen

    if st.button("Calculate Win Probability", type="primary"):
        if balls_remaining <= 0 or runs_required <= 0:
            st.warning("Match situation indicates the chase is already over.")
        else:
            prob, crr, rrr, pressure_index = predict_win_prob(
                runs_required, balls_remaining, wickets_remaining,
                runs_scored, balls_bowled, batter_skill
            )

            st.markdown("---")
            c1, c2, c3 = st.columns(3)
            c1.metric(f"{batting_team} Win Probability", f"{prob:.1%}")
            c2.metric(f"{bowling_team} Win Probability", f"{1-prob:.1%}")
            c3.metric("Pressure Index", f"{pressure_index:.1f}",
                      help="RRR minus CRR. Positive = batting team under pressure.")

            # Probability bar
            fig, ax = plt.subplots(figsize=(10, 1.5))
            ax.barh([0], [prob], color='#2ecc71', height=0.6)
            ax.barh([0], [1-prob], left=[prob], color='#e74c3c', height=0.6)
            ax.set_xlim(0, 1)
            ax.set_yticks([])
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
            ax.text(prob/2, 0, f'{batting_team}\n{prob:.0%}', ha='center', va='center',
                    color='white', fontweight='bold', fontsize=10)
            ax.text(prob + (1-prob)/2, 0, f'{bowling_team}\n{1-prob:.0%}', ha='center', va='center',
                    color='white', fontweight='bold', fontsize=10)
            for spine in ax.spines.values():
                spine.set_visible(False)
            st.pyplot(fig)

            st.markdown("#### Match State")
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Runs Required", runs_required)
            d2.metric("Balls Remaining", balls_remaining)
            d3.metric("Current RR", f"{crr:.2f}")
            d4.metric("Required RR", f"{rrr:.2f}")

# ===========================================================
# PAGE 2 — Choking Analysis
# ===========================================================
elif page == "Choking Analysis":
    st.title("📉 Team Choking Analysis")
    st.markdown(
        "**Definition:** A team 'chokes' when they reach ≥80% win probability at some point "
        "in the chase but still lose the match."
    )

    choke_sorted = choke_stats.sort_values('choke_rate', ascending=False)

    fig, ax = plt.subplots(figsize=(11, 6))
    colors = ['#e74c3c' if r > 25 else '#f39c12' if r > 15 else '#2ecc71'
              for r in choke_sorted['choke_rate']]
    bars = ax.bar(choke_sorted['batting_team'], choke_sorted['choke_rate'], color=colors)
    ax.axhline(choke_sorted['choke_rate'].mean(), color='black', linestyle='--',
               linewidth=1.2, label=f"Average: {choke_sorted['choke_rate'].mean():.1f}%")
    ax.set_ylabel("Choke Rate (%)")
    ax.set_xlabel("Team")
    plt.xticks(rotation=45, ha='right')
    for bar, val in zip(bars, choke_sorted['choke_rate']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', fontsize=9)
    ax.legend()
    st.pyplot(fig)

    st.markdown("#### Full Data")
    st.dataframe(
        choke_sorted.rename(columns={
            'batting_team': 'Team',
            'times_in_winning_position': 'Times Reached 80%+',
            'times_choked': 'Times Lost Anyway',
            'choke_rate': 'Choke Rate (%)'
        }),
        use_container_width=True,
        hide_index=True
    )

# ===========================================================
# PAGE 3 — About
# ===========================================================
else:
    st.title("About This Model")
    st.markdown("""
    ### What this is
    A ball-by-ball win probability model trained on IPL data (2008–2026), predicting the
    chasing team's probability of winning at any point in the second innings.

    ### Model
    - **Algorithm:** Random Forest Classifier
    - **Features:** Runs required, balls remaining, wickets remaining, current run rate,
      required run rate, a custom Pressure Index (RRR − CRR), and batter skill (career strike rate)
    - **Performance:** ~81% accuracy, AUC 0.90, tested on the most recent two seasons
      (held out, not used in training)
    - **Calibration:** Verified — when the model predicts 70%, teams win ~70% of the time

    ### Key Finding
    Required run rate and the custom Pressure Index together account for the majority of
    predictive power — far more than wickets remaining alone. Team identity and venue were
    tested separately and added negligible predictive value once match state was accounted for.

    ### Built by
    Prince — [https://github.com/princebawania/ipl-win-probability/tree/main]
    """)
