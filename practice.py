import pandas as pd

df = pd.read_csv("deliveries.csv")

balls = df.groupby('bowler')['over'].count()
overs = balls / 6
runs = df.groupby('bowler')['total_runs'].sum()
economy = runs / overs

# Only keep bowlers with 20+ overs
economy = economy[balls >= 120]

print(economy.sort_values().head(5))