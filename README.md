# IPL Win Probability Model

A ball-by-ball win probability model for IPL T20 matches, built end-to-end from raw data to a deployed interactive app — including a custom pressure metric, model calibration verification, and a data-driven analysis of which teams "choke" in winning positions.

**🔗 Live App:** [ipl-win-probability.streamlit.app](https://ipl-win-probability-hkx77bzv8gplut7pzijcwb.streamlit.app/)

---

## Problem Statement

Given the match state at any point in a second innings (runs required, balls remaining, wickets in hand), can we predict the chasing team's probability of winning and can we trust that probability to actually mean what it says?

This project goes beyond a standard classification exercise by:
- Engineering a custom **Pressure Index** that outperforms raw wickets/run-rate as a predictor
- Explicitly **verifying model calibration** (not just accuracy)
- Testing whether **team identity and venue** matter beyond match state (they largely don't)
- Using the model to quantify **team choking behavior** — a novel downstream application

---

## Key Findings

| Finding | Detail |
|---|---|
| **Model performance** | Random Forest — 81% accuracy, AUC 0.90 (tested on held-out 2025–26 seasons) |
| **Calibration** | Predicted vs. actual win rates differ by less than 4% across all probability bins |
| **Strongest predictor** | Required Run Rate (RRR), followed by the custom Pressure Index (RRR − CRR) |
| **Wickets matter less than expected** | Run rate pressure outweighs wickets remaining in predicting outcome |
| **Team & venue test** | Adding team identity + venue improved AUC by only 0.003 — situational pressure dominates over "who is playing" |
| **Batter skill matters** | Adding the current batter's career strike rate was the single best feature addition beyond match state |
| **Choking analysis** | Quantified which teams most often lose despite reaching 80%+ win probability |

---

##  Project Structure

```
ipl-win-probability/
├── app.py                        # Streamlit app (live predictor, choking dashboard)
├── requirements.txt              # Python dependencies
├── ipl_model.pkl                 # Trained Random Forest model
├── batter_skill_lookup.csv       # Career strike rate per batter (for app input)
├── teams_venues.json             # Dropdown options for the app
├── choke_stats.csv               # Precomputed team choke rates
└── IPL_Win_Probability.ipynb     # Full analysis notebook: cleaning → EDA → modeling → insights
```

---

##  Approach

**1. Data Cleaning** — Standardized inconsistent team names (e.g., "Royal Challengers Bangalore" vs "Royal Challengers Bengaluru"), fixed mixed-type season values and removed Super Overs.

**2. Feature Engineering** — Built match-state features (runs required, balls remaining, wickets remaining, current/required run rate) plus a custom **Pressure Index** (RRR − CRR) to capture chase difficulty in a single number.

**3. Modeling** — Compared Logistic Regression, Random Forest, and XGBoost. Random Forest performed best. Used a **time-based train/test split** (held out the most recent two seasons) rather than a random split, to simulate realistic forecasting on unseen future matches.

**4. Calibration Check** — Verified that predicted probabilities match real-world outcomes (e.g., when the model says 70%, teams actually win ~70% of the time) — a step most fresher projects skip entirely.

**5. Feature Experiments** — Tested whether team identity, venue, batter skill, and recent wicket momentum add predictive value beyond match state. Found batter skill (career strike rate) was the only meaningful addition; team/venue contributed almost nothing once situational pressure was accounted for.

**6. Choking Analysis** — Used the trained model to flag every instance a team reached ≥80% win probability, then measured how often they still lost — surfacing which franchises are statistically more prone to collapsing from winning positions.

**7. Upset Detector** — Identified the most dramatic individual match collapses, including a cross-cut with toss outcomes to find matches where teams overcame both a lost toss and a statistical deficit.

**8. Deployment** — Packaged the model into an interactive Streamlit app for live what-if scenario testing.

---

## Running Locally

```bash
git clone https://github.com/princebawania/ipl-win-probability.git
cd ipl-win-probability
pip install -r requirements.txt
streamlit run app.py
```

---

## Tech Stack

Python · pandas · scikit-learn · matplotlib · Streamlit

---

## Data Source

Ball-by-ball IPL data (2008–2026), sourced from publicly available IPL match datasets.

---

