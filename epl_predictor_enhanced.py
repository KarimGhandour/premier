"""
## Project Context and Motivation

Football (soccer) is more than just a sport—it's a global phenomenon that inspires passion and endless debate. At the heart of these discussions lies a fundamental question, which is the focus of this project:

> **Is it possible to predict the outcomes of future football matches using historical data?**

Our project, **PremierPredict**, explores this question by developing a predictive model for English Premier League matches. Beyond mere curiosity, this project is a practical application of data analysis and machine learning.

The Premier League, with its high level of competition and abundant data, is an ideal testing ground. To guide our work, we break down the main question into two sub-questions:

> 1. **What factors truly influence the outcome of a football match?**
> 2. **Given these factors, can we build a model that predicts future results more accurately than random guessing?**

By combining thorough data analysis with machine learning models, we aim to capture the subtleties of the game and turn raw statistics into reliable predictions, offering new insights into this universal sport.
"""

"""
We have deployed the predictive model developed in this project as a web application, allowing users to predict the outcomes of various matches!

#### The application is available here: [PremierPredict](https://foot-forecast.streamlit.app)
"""

"""
# Question 1: What factors actually influence the outcome of a football match?

In this section, we explore the variables that significantly impact match outcomes in the Premier League. Beyond common intuition, we aim to precisely quantify the effect of different factors on a team's probability of winning.
"""

"""
## Importing Libraries and Data

We use the `pandas` library for data analysis and manipulation.
"""

import pandas as pd  # For data manipulation and analysis

"""
We import the match data from a `.csv` file into a pandas DataFrame. This format is ideal for tabular data and allows for efficient data cleaning, transformation, and analysis.
"""

matchs = pd.read_csv('epl_2000-2025.csv')

"""
## Data Cleaning

### Renaming Columns for Clarity

The dataset contains 22 variables and 9,318 observations.

Column names are explained in the `column_def.txt` file. For clarity and reproducibility, we rename columns to more descriptive names in English. This makes the code easier to read and maintain, especially for non-French speakers or future collaborators.
"""

print(matchs.shape)
print(matchs.columns)

"""
We rename the columns using the `.rename` method with `inplace=True` to update the DataFrame directly.

**Note:** If you plan to use the original column names elsewhere, consider making a copy of the DataFrame before renaming.
"""

matchs.rename(columns={
    'Season': 'Season',
    'HomeTeam': 'Home',
    'AwayTeam': 'Away',
    'FTR': 'Result',
    'HTR': 'HalfTimeResult',
    'FTHG': 'HomeGoals',
    'FTAG': 'AwayGoals',
    'HTHG': 'HomeGoalsHT',
    'HTAG': 'AwayGoalsHT',
    'HS': 'HomeShots',
    'AS': 'AwayShots',
    'HST': 'HomeShotsOnTarget',
    'AST': 'AwayShotsOnTarget',
    'HF': 'HomeFouls',
    'AF': 'AwayFouls',
    'HC': 'HomeCorners',
    'AC': 'AwayCorners',
    'HY': 'HomeYellows',
    'AY': 'AwayYellows',
    'HR': 'HomeReds',
    'AR': 'AwayReds'
}, inplace=True)

print(matchs.columns)

"""
After renaming, we can better visualize the data in the DataFrame using `df.head()`. This helps confirm that the columns are correctly named and gives a quick look at the data structure.
"""

print(matchs.head()) 