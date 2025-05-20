def predire_matchs_matrix(top_teams, date):
    """
    Vectorized prediction for all matchups between the top teams (excluding self-matchups).
    Parameters:
      top_teams: list of team names (e.g. ['Arsenal', 'Chelsea', 'Manchester City', 'Manchester United', 'Liverpool'])
      date: a date string that will be used for all predictions
    Returns:
      A tuple (domicile_list, exterieur_list, y_pred, y_pred_prob) where domicile_list and exterieur_list are the team names for the matchups,
      y_pred are the model predictions and y_pred_prob their associated probabilities.
    """
    # List of features expected by the XGBoost model (should match the ordering used in the original predire_matchs function).
    features = [
        'DomicileCode', 'ExterieurCode', 'DomicileForme', 'ExterieurForme', 
        'DiffForme', 'FAF_VictoiresDomicile_Dom', 'FAF_Diff', 'FAF_DiffGlobal', 
        'FAF_VictoiresExterieur_Dom', 'FAF_VictoiresDomicile_Ext', 
        'FAF_VictoiresExterieur_Ext', 'SemaineSaison', 'Mois', 'AnneeSaison',
        'DomicileAvgButsMarques_Home', 'DomicileAvgButsEncaisses_Home', 
        'DomicileAvgButsMarques_Away', 'DomicileAvgButsEncaisses_Away', 
        'ExterieurAvgButsMarques_Home', 'ExterieurAvgButsEncaisses_Home', 
        'ExterieurAvgButsMarques_Away', 'ExterieurAvgButsEncaisses_Away', 
        'DiffButsDomicile', 'DiffButsExterieur', 'DiffButs', 'DiffButsGlobal',
        'FAF_Nul_Domicile_Dom', 'FAF_Nul_Domicile_Ext', 'FAF_DiffNul'
    ]

    # Convert the input date into a pandas Timestamp
    date_pd = pd.to_datetime(date)

    # Create all matchups between top teams, excluding same team matchups
    domicile_list = [team_i for team_i in top_teams for team_j in top_teams if team_i != team_j]
    exterieur_list = [team_j for team_i in top_teams for team_j in top_teams if team_i != team_j]
    n = len(domicile_list)

    # Initialize a dictionary to build the DataFrame vectorized
    data = {}
    # For all features, initialize with zeros
    for col in features:
        data[col] = [0] * n

    # Fill in the vectorized fields
    data['Date'] = [date_pd] * n
    data['DomicileCode'] = [code_equipe.get(team) for team in domicile_list]
    data['ExterieurCode'] = [code_equipe.get(team) for team in exterieur_list]
    data['Mois'] = [date_pd.month] * n
    # Determine AnneeSaison based on the date
    data['AnneeSaison'] = [date_pd.year - 1 if date_pd > pd.Timestamp(date_pd.year, 8, 5) else date_pd.year] * n
    # Compute SemaineSaison using vectorized list comprehension
    data['SemaineSaison'] = [int(((date_pd - pd.Timestamp(year=data['AnneeSaison'][i], month=8, day=5)).days // 7) + 1) for i in range(n)]

    # Create the DataFrame for predictions
    new_rows_df = pd.DataFrame(data)

    # Ensure the DataFrame has the correct column order
    X_pred = new_rows_df[features]
    
    # Run predictions using the XGBoost model
    y_pred = xgb_model.predict(X_pred)
    y_pred_prob = xgb_model.predict_proba(X_pred)

    return domicile_list, exterieur_list, y_pred, y_pred_prob
