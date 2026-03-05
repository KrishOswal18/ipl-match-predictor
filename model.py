import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("deliveries.csv")

# ── Rebuild match dataset ────────────────────────────────────
match_scores = df.groupby(['match_id', 'inning', 'batting_team'])['total_runs'].sum().reset_index()
inn1 = match_scores[match_scores['inning'] == 1][['match_id', 'batting_team', 'total_runs']]
inn2 = match_scores[match_scores['inning'] == 2][['match_id', 'batting_team', 'total_runs']]
inn1.columns = ['match_id', 'team1', 'score1']
inn2.columns = ['match_id', 'team2', 'score2']
matches = pd.merge(inn1, inn2, on='match_id')
matches['winner'] = matches.apply(lambda r: r['team2'] if r['score2'] > r['score1'] else r['team1'], axis=1)
matches['team1_won'] = (matches['winner'] == matches['team1']).astype(int)
wickets = df.groupby(['match_id', 'inning'])['is_wicket'].sum().reset_index()
w1 = wickets[wickets['inning'] == 1][['match_id', 'is_wicket']].rename(columns={'is_wicket': 'wickets1'})
w2 = wickets[wickets['inning'] == 2][['match_id', 'is_wicket']].rename(columns={'is_wicket': 'wickets2'})
matches = pd.merge(matches, w1, on='match_id')
matches = pd.merge(matches, w2, on='match_id')

# ── Encode team names to numbers ─────────────────────────────
le = LabelEncoder()
all_teams = pd.concat([matches['team1'], matches['team2']])
le.fit(all_teams)
matches['team1_enc'] = le.transform(matches['team1'])
matches['team2_enc'] = le.transform(matches['team2'])

# ── Better features ──────────────────────────────────────────

# Run rate in first innings
matches['run_rate1'] = matches['score1'] / 20

# Historical win rate per team
win_counts = matches['winner'].value_counts()
total_counts = pd.concat([matches['team1'], matches['team2']]).value_counts()
matches['team1_winrate'] = matches['team1'].map(win_counts) / matches['team1'].map(total_counts)
matches['team2_winrate'] = matches['team2'].map(win_counts) / matches['team2'].map(total_counts)
matches[['team1_winrate', 'team2_winrate']] = matches[['team1_winrate', 'team2_winrate']].fillna(0.5)

# ── Features and target ──────────────────────────────────────
X = matches[['team1_enc', 'team2_enc', 'score1', 'wickets1',
             'run_rate1', 'team1_winrate', 'team2_winrate']]
y = matches['team1_won']

# ── Train/test split ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── Train Random Forest ──────────────────────────────────────
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ── Results ──────────────────────────────────────────────────
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {acc*100:.1f}%")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

# ── Feature importance ───────────────────────────────────────
feat_names = ['team1', 'team2', 'score1', 'wickets1']
importances = model.feature_importances_
print("\nWhat matters most for prediction:")
for f, i in sorted(zip(feat_names, importances), key=lambda x: -x[1]):
    print(f"  {f}: {i*100:.1f}%")

import pickle

# Save the model and encoder so Streamlit can use them
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print("\nModel saved!")