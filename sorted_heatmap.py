import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ----- SETTINGS & ELO PARAMETERS -----
big6 = [
    'Arsenal',
    'Chelsea',
    'Manchester United',
    'Liverpool',
    'Manchester City',
    'Tottenham'
]

team_name_map = {
    'Man United': 'Manchester United',
    'Man Utd': 'Manchester United',
    'Manchester Utd': 'Manchester United',
    'Man City': 'Manchester City'
}

compute_start_year = 2010
end_year = 2024
initial_elo = 2000
K = 30
home_advantage = 100

# ----- LOAD & PREPARE DATA -----
df = pd.read_csv('epl_2000-2025.csv')
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df['Year'] = df['Date'].dt.year

# Standardize team names
df['HomeTeam'] = df['HomeTeam'].apply(lambda x: team_name_map.get(x, x))
df['AwayTeam'] = df['AwayTeam'].apply(lambda x: team_name_map.get(x, x))

# Filter data for 2010–2024 and only head-to-head matches among Big 6 teams
df = df[(df['Year'] >= compute_start_year) & (df['Year'] <= end_year)]
df_big6 = df[(df['HomeTeam'].isin(big6)) & (df['AwayTeam'].isin(big6))].copy()
df_big6 = df_big6.sort_values(['Date', 'HomeTeam', 'AwayTeam'])

# ----- COMPUTE FINAL ELO RATINGS (2024) -----
elo_ratings = {team: initial_elo for team in big6}
for idx, row in df_big6.iterrows():
    home_team = row['HomeTeam']
    away_team = row['AwayTeam']
    home_rating = elo_ratings[home_team]
    away_rating = elo_ratings[away_team]
    
    # Compute expected scores (including home advantage for home team)
    exp_home = 1 / (1 + 10 ** ((away_rating - (home_rating + home_advantage)) / 400))
    exp_away = 1 - exp_home
    
    # Determine actual match result
    if row['FTHG'] > row['FTAG']:
        score_home = 1
    elif row['FTHG'] < row['FTAG']:
        score_home = 0
    else:
        score_home = 0.5
    score_away = 1 - score_home
    
    # Update ELO ratings using the official formula
    elo_ratings[home_team] = home_rating + K * (score_home - exp_home)
    elo_ratings[away_team] = away_rating + K * (score_away - exp_away)

# Sort teams from best to worst by final ELO
sorted_teams = sorted(big6, key=lambda t: elo_ratings[t], reverse=True)

# ----- CREATE WIN PROBABILITY MATRIX (neutral ground) -----
matrix = pd.DataFrame(index=sorted_teams, columns=sorted_teams, dtype=float)
for t_i in sorted_teams:
    for t_j in sorted_teams:
        if t_i == t_j:
            matrix.loc[t_i, t_j] = np.nan
        else:
            R_i = elo_ratings[t_i]
            R_j = elo_ratings[t_j]
            prob = 1 / (1 + 10 ** ((R_j - R_i) / 400))
            matrix.loc[t_i, t_j] = prob * 100

# ----- PLOT HEATMAP WITH ADJUSTED LABEL ORIENTATIONS, SMALLER FONT, AND NO GRIDLINES -----
plt.figure(figsize=(6, 6))
sns.set_style('white')  # Remove gridlines
ax = sns.heatmap(matrix, annot=True, fmt=".1f", cmap="coolwarm",
                 cbar_kws={'label': 'Win Probability (%)'},
                 vmin=0, vmax=100)
# X-axis labels: horizontal, bold, larger font (approx. 35% larger than 8 -> 11)
plt.xticks(rotation=0, fontsize=11, fontweight='bold')
# Y-axis labels: vertical, bold, larger font
plt.yticks(rotation=90, fontsize=11, fontweight='bold')

plt.title("Big 6 Head-to-Head Win Probabilities (2024) [Sorted Teams]", fontsize=14)
plt.xlabel("Opponent", fontsize=12)
plt.ylabel("Team", fontsize=12)
plt.tight_layout()
plt.show()

print("Final ELO Ratings (2024), sorted high to low:")
for t in sorted_teams:
    print(f"{t}: {elo_ratings[t]:.1f}")