import pandas as pd

df = pd.read_csv("deliveries.csv")

# ── STEP 1: Build match-level features ──────────────────────
match_scores = df.groupby(['match_id', 'inning', 'batting_team'])['total_runs'].sum().reset_index()

# Separate innings 1 and 2
inn1 = match_scores[match_scores['inning'] == 1][['match_id', 'batting_team', 'total_runs']]
inn2 = match_scores[match_scores['inning'] == 2][['match_id', 'batting_team', 'total_runs']]

inn1.columns = ['match_id', 'team1', 'score1']
inn2.columns = ['match_id', 'team2', 'score2']

matches = pd.merge(inn1, inn2, on='match_id')

# ── STEP 2: Who won? (team batting second wins if score2 > score1) ──
matches['winner'] = matches.apply(
    lambda r: r['team2'] if r['score2'] > r['score1'] else r['team1'], axis=1
)
matches['team1_won'] = (matches['winner'] == matches['team1']).astype(int)

# ── STEP 3: Wickets per innings ──────────────────────────────
wickets = df.groupby(['match_id', 'inning'])['is_wicket'].sum().reset_index()
w1 = wickets[wickets['inning'] == 1][['match_id', 'is_wicket']].rename(columns={'is_wicket': 'wickets1'})
w2 = wickets[wickets['inning'] == 2][['match_id', 'is_wicket']].rename(columns={'is_wicket': 'wickets2'})

matches = pd.merge(matches, w1, on='match_id')
matches = pd.merge(matches, w2, on='match_id')

print(matches.head(10))
print(f"\nShape: {matches.shape}")
print(f"\nTeam1 win rate: {matches['team1_won'].mean()*100:.1f}%")