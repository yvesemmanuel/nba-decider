from datetime import datetime
import os
import json
import pickle

with open('./services/data/seasons.json') as file:
    seasons = json.load(file)


def load_predictors(game_date: datetime) -> list:
    game_month, game_year = game_date.month, game_date.year
    predictors_with_date = []

    for season_year in range(min(seasons), game_year + 1):
        season_folder = './services/models/' + str(season_year)

        if os.path.exists(season_folder):
            for filename in os.listdir(season_folder):
                if filename.endswith('.pkl'):
                    _, month, year_part = filename.split('_')
                    year = year_part.split('.')[0]

                    file_year = int(year)
                    file_month = int(month)

                    if file_year < game_year or (file_year == game_year and file_month <= game_month):
                        file_path = os.path.join(season_folder, filename)
                        with open(file_path, 'rb') as file:
                            predictor = pickle.load(file)
                            predictors_with_date.append(((file_year, file_month), predictor))

    # Sort the list of tuples based on the dates
    predictors_with_date.sort(key=lambda x: x[0])

    # Extract the sorted predictors
    predictors = [predictor for _, predictor in predictors_with_date]

    return predictors


def calculate_weighted_probas(probabilities, general_weight=1, last_weight=3):
    weights = [general_weight] * (len(probabilities) - 1) + [last_weight]
    weighted_sum = sum(weight * proba for weight, proba in zip(weights, probabilities)) / sum(weights)

    return weighted_sum
