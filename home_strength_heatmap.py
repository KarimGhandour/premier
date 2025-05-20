import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import linregress

# ----- Data Loading and Preparation -----
# Load overall win percentage dataset (assumes file was generated previously)
win_df = pd.read_csv('big6_win_percentage_by_year.csv')
win_df = win_df[(win_df['Year'] >= 2014) & (win_df['Year'] <= 2024)].sort_values(['Team', 'Year'])
if 'Rolling_Win_Pct' not in win_df.columns:
    win_df['Rolling_Win_Pct'] = win_df.groupby('Team')['Win_Percentage']\
                                      .transform(lambda x: x.rolling(window=3, min_periods=1).mean())

# Load home advantage dataset (assumes file was generated previously)
home_df = pd.read_csv('big6_home_advantage_by_year.csv')
home_df = home_df[(home_df['Year'] >= 2014) & (home_df['Year'] <= 2024)].sort_values(['Team', 'Year'])
if 'Rolling_Home_Adv' not in home_df.columns:
    home_df['Rolling_Home_Adv'] = home_df.groupby('Team')['Home_Advantage']\
                                        .transform(lambda x: x.rolling(window=3, min_periods=1).mean())

# Merge datasets on Team and Year
merged_df = pd.merge(win_df[['Team', 'Year', 'Rolling_Win_Pct']],
                     home_df[['Team', 'Year', 'Rolling_Home_Adv']],
                     on=['Team', 'Year'], how='inner')

# ----- Compute Moving Window Correlations -----
# Define moving window size (e.g., 3-year window)
window_size = 3

# Compute moving window correlations (R²) for each team
corr_results = []
for team in merged_df['Team'].unique():
    team_data = merged_df[merged_df['Team'] == team].sort_values('Year').reset_index(drop=True)
    for i in range(len(team_data) - window_size + 1):
        window_data = team_data.iloc[i:i + window_size]
        x = window_data['Rolling_Home_Adv'].values
        y = window_data['Rolling_Win_Pct'].values
        if len(x) == window_size and len(y) == window_size:
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            r_squared = r_value ** 2
            # Use the median year of the window as the representative year
            mid_year = int(window_data['Year'].median())
            corr_results.append({'Team': team, 'Year': mid_year, 'R_squared': r_squared})

# Convert the results into a DataFrame and pivot to create a heatmap-friendly format
corr_df = pd.DataFrame(corr_results)
heatmap_data = corr_df.pivot(index='Team', columns='Year', values='R_squared')

# ----- Visualization -----
plt.figure(figsize=(10, 6))
# Using the 'YlOrRd' colormap: warmer colours indicate higher correlations
sns.heatmap(heatmap_data, annot=True, cmap='YlOrRd', fmt='.2f', cbar_kws={'label': 'R Squared'})
plt.title('Moving Window R² of Home Advantage vs. Overall Win Percentage', fontsize=16)
plt.xlabel('Year (Mid of 3-Year Window)', fontsize=12)
plt.ylabel('Team', fontsize=12)
plt.tight_layout()
plt.show()thi