"""Miscellaneous functions."""

import pandas as pd
import numpy as np

def get_numeric_columns(df: pd.DataFrame):
    """
    Returns a list of column names that contain numerical data in the given DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to search for numerical columns in.
        
    Returns:
    --------
    List of strings
        A list of column names that contain numerical data.
    """
    numeric_cols = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
    return numeric_cols


def get_categorical_columns(df: pd.DataFrame):
    """
    Returns a list of column names that contain categorical data in the given DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to search for categorical columns in.
        
    Returns:
    --------
    List of strings
        A list of column names that contain categorical data.
    """
    categorical_cols = []
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            categorical_cols.append(col)
    return categorical_cols


def get_matchups(df: pd.DataFrame):
    """
    Extracts home and away teams from each matchup in a DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the matchups.
        
    Returns:
    --------
    Two lists containing home teams and away teams in each matchup.
    """
    home_teams = []
    away_teams = []

    for matchup in df["MATCHUP"]:
        if "@" in matchup:
            away, home = matchup.split(" @ ")
        elif "vs" in matchup:
            home, away = matchup.split(" vs. ")
        else:
            home, away = np.nan, np.nan
        home_teams.append(home)
        away_teams.append(away)

    return home_teams, away_teams
