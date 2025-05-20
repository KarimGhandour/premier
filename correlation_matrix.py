import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ----- Data Preparation -----

# Load overall win percentage dataset (assumes file generated previously)
win_df = pd.read_csv('big6_win_percentage_by_year.csv')
# Filter for the desired period (2014 to 2024) and sort for rolling calculations
win_df = win_df[(win_df['Year'] >= 2014) & (win_df['Year'] <= 2024)].sort_values(['Team', 'Year'])
# Compute 3-year rolling average for overall win percentage
win_df['Rolling_Win_Pct'] = win_df.groupby('Team')['Win_Percentage'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())

# Load home advantage dataset (assumes you have file generated previously)
home_df = pd.read_csv('big6_home_advantage_by_year.csv')
# Filter for the same period and sort
home_df = home_df[(home_df['Year'] >= 2014) & (home_df['Year'] <= 2024)].sort_values(['Team', 'Year'])
# Compute 3-year rolling average for home advantage
home_df['Rolling_Home_Adv'] = home_df.groupby('Team')['Home_Advantage'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())

# Merge the two datasets on Team and Year
merged_df = pd.merge(win_df[['Team', 'Year', 'Rolling_Win_Pct']], home_df[['Team', 'Year', 'Rolling_Home_Adv']], on=['Team', 'Year'], how='inner')

# Compute the correlation matrix between the rolling metrics
corr_matrix = merged_df[['Rolling_Win_Pct', 'Rolling_Home_Adv']].corr()

# ----- Visualization -----

sns.set(style="whitegrid")
fig, axes = plt.subplots(3, 1, figsize=(12, 18))

# Panel A: Rolling Overall Win Percentage
sns.lineplot(data=merged_df, x='Year', y='Rolling_Win_Pct', hue='Team', marker='o', ax=axes[0], linewidth=2.5)
axes[0].set_title('Rolling Overall Win Percentage (3-Year Rolling Avg)', fontsize=16)
axes[0].set_xlabel('')
axes[0].set_ylabel('Rolling Win %')
axes[0].set_ylim(0.2, 0.82)
axes[0].set_xticks(list(range(2014, 2025)))  # 1-year increments

# Panel B: Rolling Home Advantage
sns.lineplot(data=merged_df, x='Year', y='Rolling_Home_Adv', hue='Team', marker='o', ax=axes[1], linewidth=2.5)
axes[1].set_title('Rolling Home Advantage (3-Year Rolling Avg)', fontsize=16)
axes[1].set_xlabel('')
axes[1].set_ylabel('Rolling Home Win %')
axes[1].set_ylim(0.2, 0.82)
axes[1].set_xticks(list(range(2014, 2025)))

# Panel C: Correlation Heatmap
sns.heatmap(corr_matrix, annot=True, cmap='Blues', ax=axes[2], cbar=True, fmt='.2f')
axes[2].set_title('Correlation Between Rolling Win % and Rolling Home Advantage', fontsize=16)

plt.tight_layout()
plt.show()