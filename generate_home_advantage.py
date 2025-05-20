import pandas as pd
import os

# Print current working directory to verify where the file will be saved.
print("Current working directory:", os.getcwd())

# Define the Big 6 teams (adjust names if needed)
big6 = ['Arsenal', 'Chelsea', 'Manchester United', 'Liverpool', 'Manchester City', 'Tottenham']

# Load the original CSV with match data
df = pd.read_csv('epl_2000-2025.csv')

# Convert Date to datetime and extract Year
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df['Year'] = df['Date'].dt.year

# Filter data for the desired period (e.g., 2014 to 2024)
df = df[(df['Year'] >= 2014) & (df['Year'] <= 2024)]

# Filter to include only matches where the home team is in the Big 6
df_home = df[df['HomeTeam'].isin(big6)].copy()

# Define a function to get match result for the home team
def get_home_result(row):
    if row['FTHG'] > row['FTAG']:
        return 'Win'
    elif row['FTHG'] < row['FTAG']:
        return 'Loss'
    else:
        return 'Draw'

df_home['Result'] = df_home.apply(get_home_result, axis=1)

# Group by HomeTeam and Year to compute home wins and matches
home_stats = df_home.groupby(['HomeTeam', 'Year']).agg(
    Home_Matches=('Result', 'count'),
    Home_Wins=('Result', lambda x: (x == 'Win').sum())
).reset_index()

# Compute Home Advantage as Home_Wins / Home_Matches
home_stats['Home_Advantage'] = home_stats['Home_Wins'] / home_stats['Home_Matches']

# Rename 'HomeTeam' to 'Team' for consistency
home_stats.rename(columns={'HomeTeam': 'Team'}, inplace=True)

# Save the CSV file in the current directory
home_stats.to_csv('big6_home_advantage_by_year.csv', index=False)
print("big6_home_advantage_by_year.csv has been generated.")