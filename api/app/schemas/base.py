from pydantic import BaseModel, validator
from datetime import datetime
from datetime import datetime, date


class Matchup(BaseModel):
    home_name: str
    away_name: str

class PredictInput(BaseModel):
    date: date  # Change the type annotation here
    matchup: Matchup

    @validator('date', pre=False)
    def parse_date(cls, value):
        if isinstance(value, date):
            return value
        try:
            return datetime.strptime(value, '%Y-%m-%d').date()
        except ValueError:
            raise ValueError('Invalid date format, expected YYYY-MM-DD')

class Odds(BaseModel):
    home_odd: float
    away_odd: float

class GambleInput(BaseModel):
    predict_input: PredictInput
    odds: Odds
