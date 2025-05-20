import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

# ----- Data Preparation -----

# Load overall win percentage dataset (assumes you previously generated this file)
win_df = pd.read_csv('big6_win_percentage_by_year.csv')
# Filter for the period 2014 to 2024 and sort by Team and Year
win_df = win_df[(win_df['Year'] >= 2014) & (win_df['Year'] <= 2024)].sort_values(['Team', 'Year'])
# Compute a 3-year rolling average for overall win percentage
win_df['Rolling_Win_Pct'] = win_df.groupby('Team')['Win_Percentage'] \
                                  .transform(lambda x: x.rolling(window=3, min_periods=1).mean())

# Load home advantage dataset (assumes you generated big6_home_advantage_by_year.csv)
home_df = pd.read_csv('big6_home_advantage_by_year.csv')
# Filter for the same period and sort by Team and Year
home_df = home_df[(home_df['Year'] >= 2014) & (home_df['Year'] <= 2024)].sort_values(['Team', 'Year'])
# Compute a 3-year rolling average for home advantage
home_df['Rolling_Home_Adv'] = home_df.groupby('Team')['Home_Advantage'] \
                                    .transform(lambda x: x.rolling(window=3, min_periods=1).mean())

# Merge the two datasets on Team and Year so we can compare metrics
merged_df = pd.merge(win_df[['Team', 'Year', 'Rolling_Win_Pct']], 
                     home_df[['Team', 'Year', 'Rolling_Home_Adv']], 
                     on=['Team', 'Year'], how='inner')

# ----- Compute Home Strength (R² from regression) -----

# For each team, perform linear regression:
#   - Independent variable: Rolling_Home_Adv 
#   - Dependent variable: Rolling_Win_Pct
# Compute R² to measure how much home performance explains overall win rate
teams = merged_df['Team'].unique()
results = []
for team in teams:
    data = merged_df[merged_df['Team'] == team]
    x = data['Rolling_Home_Adv']
    y = data['Rolling_Win_Pct']
    # Perform linear regression; r_value is the correlation coefficient
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    r_squared = r_value ** 2
    results.append({'Team': team, 'R_squared': r_squared})

results_df = pd.DataFrame(results)
# Sort teams by R² (explained variance) descending
results_df = results_df.sort_values('R_squared', ascending=False)

# ----- Visualization -----

plt.figure(figsize=(8, 6))
# Create a horizontal bar chart to visualize the R² values for each team
plt.barh(results_df['Team'], results_df['R_squared'], color='skyblue')
plt.xlabel('R Squared (Explained Variance)', fontsize=12)
plt.title('Home Advantage Explains Historical Win Rate (R²) by Team', fontsize=14)
plt.xlim(0, 1)  # R² values range from 0 to 1
plt.tight_layout()
plt.show()