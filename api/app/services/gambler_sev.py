from schemas.base import (
    PredictInput,
    GambleInput
)
from services.utils import (
    load_predictors, 
    calculate_weighted_probas
)
from typing import Tuple
import json

class GambleService():
    def __init__(self) -> None:
        with open('./services/data/seasons.json') as file:
            self.seasons = json.load(file)

        with open('./services/data/team_mapping_id.json') as file:
            self.team_mapping_id = json.load(file)

    def predict(self, input: PredictInput, home_win_threshold: float = 0.6):
        prophet_probas = []

        home_id = self.team_mapping_id[input.matchup.home_name]
        away_id = self.team_mapping_id[input.matchup.away_name]

        for prophet in load_predictors(input.date):
            _, proba = prophet.predict(home_id, away_id)
            prophet_probas.append(proba)

        ga_proba = calculate_weighted_probas(prophet_probas)

        return {
            'home_win_proba': ga_proba,
            'pred': int(ga_proba >= home_win_threshold)
        }
    
    def inverse(self, num: float) -> float:
        return 1 / num
    
    def remove_juice(self, implied_home_proba: float, implied_away_proba: float) -> Tuple[float, float]:
        overround = implied_home_proba + implied_away_proba

        return implied_home_proba / overround, implied_away_proba / overround

    def gamble(self, input: GambleInput):
        pred_output = self.predict(input.predict_input)
        home_win_proba = pred_output['home_win_proba']
        away_win_proba = 1 - home_win_proba

        fair_home_odd = self.inverse(home_win_proba)
        fair_away_odd = self.inverse(away_win_proba)

        implied_home_proba = self.inverse(input.odds.home_odd)
        implied_away_proba = self.inverse(input.odds.away_odd)

        book_home_proba, book_away_proba = self.remove_juice(implied_home_proba, implied_away_proba)

        home_win_value_0 = home_win_proba / book_home_proba
        away_win_value_0 = away_win_proba / book_away_proba
        is_home_win_valuable_0 = home_win_value_0 > 1.1
        is_away_win_valuable_0 = away_win_value_0 > 1.1

        home_win_value_1 = home_win_proba / implied_home_proba
        away_win_value_1 = away_win_proba / implied_away_proba
        is_home_win_valuable_1 = home_win_value_1 > 1.1
        is_away_win_valuable_1 = away_win_value_1 > 1.1

        return {
            'fair_line': {
                'home': fair_home_odd,
                'away': fair_away_odd
            },
            'probas': {
                'without_juice': {
                    'home': home_win_value_0,
                    'away': away_win_value_0
                }
            },
            'bet_decision_weak': {
                'home': {
                    'value_0': home_win_value_0,
                    'valuable': is_home_win_valuable_0
                },
                'away': {
                    'value': away_win_value_0,
                    'valuable': is_away_win_valuable_0
                }
            },
            'bet_decision_strong': {
                'home': {
                    'value_0': home_win_value_1,
                    'valuable': is_home_win_valuable_1
                },
                'away': {
                    'value': away_win_value_1,
                    'valuable': is_away_win_valuable_1
                }
            }
        }
